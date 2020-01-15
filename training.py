import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import os, sys
import time
import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support
import matplotlib.pyplot as plt

sys.path.append('./auxiliary/')
from auxiliary.model import PoseEstimator, BaselineEstimator
from auxiliary.dataset import Pascal3D, ShapeNet, Pix3D
from auxiliary.utils import KaiMingInit, save_checkpoint, load_checkpoint, AverageValueMeter, \
    get_pred_from_cls_output, plot_loss_fig, plot_acc_fig, rotation_acc
from auxiliary.loss import CELoss, DeltaLoss
from evaluation import val

# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()

# network training procedure settings
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of optimizer')
parser.add_argument('--decrease', type=int, default=100, help='epoch to decrease')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--print_freq', type=int, default=50, help='frequence of output print')

# model hyper-parameters
parser.add_argument('--model', type=str, default=None, help='optional reload model path')
parser.add_argument('--img_feature_dim', type=int, default=1024, help='feature dimension for images')
parser.add_argument('--shape_feature_dim', type=int, default=256, help='feature dimension for shapes')
parser.add_argument('--bin_size', type=int, default=15, help='bin size for the euler angle classification')

# dataset settings
parser.add_argument('--dataset', type=str, default=None,
                    choices=['ObjectNet3D', 'Pascal3D', 'ShapeNetCore'], help='dataset')
parser.add_argument('--shape_dir', type=str, default='Renders_semi_sphere',
                    choices=['Renders_semi_sphere', 'pointcloud'], help='subdirectory conatining the shape')
parser.add_argument('--shape', type=str, default='MultiView',
                    choices=['MultiView', 'PointCloud'], help='shape representation')
parser.add_argument('--view_num', type=int, default=12, help='number of render images used in each sample')
parser.add_argument('--tour', type=int, default=2, help='elevation tour for randomized references')
parser.add_argument('--novel', action='store_true', help='whether to test on novel cats')
parser.add_argument('--keypoint', action='store_true', help='whether to use only training samples with anchors')

# canonical view randomization as data augmentation
parser.add_argument('--random', action='store_true', help='activate random canonical view data augmentation')
parser.add_argument('--random_range', type=int, default=0, help='variation range for randomized references')

opt = parser.parse_args()
print(opt)
# ========================================================== #


# ==================RANDOM SEED SETTING===================== #
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
# ========================================================== #


# =================CREATE DATASET=========================== #
root_dir = os.path.join('data', opt.dataset)
annotation_file = '{}.txt'.format(opt.dataset)

if opt.dataset == 'ObjectNet3D':
    test_cats = ['bed', 'bookshelf', 'calculator', 'cellphone', 'computer', 'door', 'filing_cabinet', 'guitar', 'iron',
                 'knife', 'microwave', 'pen', 'pot', 'rifle', 'shoe', 'slipper', 'stove', 'toilet', 'tub', 'wheelchair']
    dataset_train = Pascal3D(train=True, root_dir=root_dir, annotation_file=annotation_file,
                             cat_choice=test_cats, keypoint=opt.keypoint, novel=opt.novel,
                             shape=opt.shape, shape_dir=opt.shape_dir, view_num=opt.view_num, tour=opt.tour,
                             random_range=opt.random_range, random=opt.random)
    dataset_eval = Pascal3D(train=False, root_dir=root_dir, annotation_file=annotation_file,
                            cat_choice=test_cats, keypoint=opt.keypoint, random=False, novel=opt.novel,
                            shape=opt.shape, shape_dir=opt.shape_dir, view_num=opt.view_num, tour=opt.tour)
elif opt.dataset == 'Pascal3D':
    test_cats = ['bus', 'motorbike'] if opt.novel else None
    dataset_train = Pascal3D(train=True, root_dir=root_dir, annotation_file=annotation_file,
                             cat_choice=test_cats, novel=opt.novel,
                             shape=opt.shape, shape_dir=opt.shape_dir, view_num=opt.view_num, tour=opt.tour,
                             random=opt.random, random_range=opt.random_range)
    dataset_eval = Pascal3D(train=False, root_dir=root_dir, annotation_file=annotation_file,
                            shape=opt.shape, shape_dir=opt.shape_dir, view_num=opt.view_num, tour=opt.tour,
                            random=False, cat_choice=test_cats, novel=opt.novel)
elif opt.dataset == 'ShapeNetCore':
    # train on synthetic data and evaluate on real data
    bg_dir = os.path.join('data', 'SUN')
    test_root_dir = os.path.join('data', 'Pix3D')
    test_annotation_file = 'Pix3D.txt'
    test_cats = ['2818832', '2871439', '2933112', '3001627', '4256520', '4379243']
    
    dataset_train = ShapeNet(train=True, root_dir=root_dir, annotation_file=annotation_file, bg_dir=bg_dir,
                             shape=opt.shape, random=opt.random, cat_choice=test_cats, novel=opt.novel,
                             view_num=opt.view_num, tour=opt.tour, random_range=opt.random_range)
    dataset_eval = Pix3D(root_dir=test_root_dir, annotation_file=test_annotation_file,
                         shape=opt.shape, view_num=opt.view_num, tour=opt.tour)
else:
    sys.exit(0)
    
train_loader = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, drop_last=True)
eval_loader = DataLoader(dataset_eval, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, drop_last=True)
# ========================================================== #


# ================CREATE NETWORK============================ #
azi_classes, ele_classes, inp_classes = int(360 / opt.bin_size), int(180 / opt.bin_size), int(360 / opt.bin_size)

if opt.shape is None:
    model = BaselineEstimator(img_feature_dim=opt.img_feature_dim,
                              azi_classes=azi_classes, ele_classes=ele_classes, inp_classes=inp_classes)
else:
    model = PoseEstimator(shape=opt.shape, shape_feature_dim=opt.shape_feature_dim, img_feature_dim=opt.img_feature_dim,
                          azi_classes=azi_classes, ele_classes=ele_classes, inp_classes=inp_classes, view_num=opt.view_num)

model.cuda()
if opt.model is not None:
    load_checkpoint(model, opt.model)
else:
    model.apply(KaiMingInit)
# ========================================================== #


# ================CREATE OPTIMIZER AND LOSS================= #
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0.0005)
lrScheduler = optim.lr_scheduler.MultiStepLR(optimizer, [opt.decrease], gamma=0.1)
criterion_azi = CELoss(360)
criterion_ele = CELoss(180)
criterion_inp = CELoss(360)
criterion_reg = DeltaLoss(opt.bin_size)
# ========================================================== #


# =============DEFINE stuff for logs ======================= #
# write basic information into the log file
training_mode = 'baseline_{}'.format(opt.dataset) if opt.shape is None else '{}_{}'.format(opt.shape, opt.dataset)
if opt.novel:
    training_mode = '{}_novel'.format(training_mode)
result_path = os.path.join(os.getcwd(), 'result', training_mode)
if not os.path.exists(result_path):
    os.makedirs(result_path)
logname = os.path.join(result_path, 'training_log.txt')
with open(logname, 'a') as f:
    f.write(str(opt) + '\n' + '\n')
    f.write('training set: ' + str(len(dataset_train)) + '\n')
    f.write('evaluation set: ' + str(len(dataset_eval)) + '\n')

# arrays for saving the losses and accuracies
losses = np.zeros((opt.n_epoch, 2))  # training loss and validation loss
accuracies = np.zeros((opt.n_epoch, 2))  # train and val accuracy for classification and viewpoint estimation
# ========================================================== #


# =================== DEFINE TRAIN ========================= #
def train(data_loader, model, bin_size, shape, criterion_azi, criterion_ele, criterion_inp, criterion_reg, optimizer):
    train_loss = AverageValueMeter()
    train_acc_rot = AverageValueMeter()

    model.train()

    data_time = AverageValueMeter()
    batch_time = AverageValueMeter()
    end = time.time()
    for i, data in enumerate(data_loader):
        # load data and label
        if shape is not None:
            im, shapes, label = data
            im, shapes, label = im.cuda(), shapes.cuda(), label.cuda()
        else:
            im, label = data
            im, label = im.cuda(), label.cuda()
        data_time.update(time.time() - end)

        # forward pass
        out = model(im) if shape is None else model(im, shapes)

        # compute losses and update the meters
        loss_azi = criterion_azi(out[0], label[:, 0])
        loss_ele = criterion_ele(out[1], label[:, 1])
        loss_inp = criterion_inp(out[2], label[:, 2])
        loss_reg = criterion_reg(out[3], out[4], out[5], label.float())
        loss = loss_azi + loss_ele + loss_inp + loss_reg
        train_loss.update(loss.item(), im.size(0))

        # compute rotation matrix accuracy
        preds = get_pred_from_cls_output([out[0], out[1], out[2]])
        for n in range(len(preds)):
            pred_delta = out[n + 3]
            delta_value = pred_delta[torch.arange(pred_delta.size(0)), preds[n].long()].tanh() / 2
            preds[n] = (preds[n].float() + delta_value + 0.5) * bin_size
        acc_rot = rotation_acc(torch.cat((preds[0].unsqueeze(1), preds[1].unsqueeze(1), preds[2].unsqueeze(1)), 1),
                               label.float())
        train_acc_rot.update(acc_rot.item(), im.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure bacth time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % opt.print_freq == 0:
            print("\tEpoch %3d --- Iter [%d/%d] Train loss: %.2f || Train accuracy: %.2f" %
                  (epoch, i + 1, len(data_loader), train_loss.avg, train_acc_rot.avg))
            print("\tData loading time: %.2f (%.2f)-- Batch time: %.2f (%.2f)\n" %
                  (data_time.val, data_time.avg, batch_time.val, batch_time.avg))

    return [train_loss.avg, train_acc_rot.avg]
# ========================================================== #


# =============BEGIN OF THE LEARNING LOOP=================== #
# initialization
best_acc = 0.

for epoch in range(opt.n_epoch):
    # update learning rate
    lrScheduler.step()

    # train
    train_loss, train_acc_rot = train(train_loader, model, opt.bin_size, opt.shape,
                                      criterion_azi, criterion_ele, criterion_inp, criterion_reg, optimizer)

    # evaluate
    eval_loss, eval_acc_rot, _, _ = val(eval_loader, model, opt.bin_size, opt.shape,
                                        criterion_azi, criterion_ele, criterion_inp, criterion_reg)

    # update best_acc and save checkpoint
    is_best = eval_acc_rot > best_acc
    best_acc = max(best_acc, eval_acc_rot)
    losses[epoch, :] = [train_loss, eval_loss]
    accuracies[epoch, :] = [train_acc_rot, eval_acc_rot]
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
        'losses': losses,
        'accuracies': accuracies
    }, is_best, os.path.join(result_path, 'checkpoint.pth'), result_path)

    # save losses and accuracies into log file
    with open(logname, 'a') as f:
        text = str('Epoch: %03d || train_loss %.2f -- val_loss %.2f || train_acc %.2f -- val_acc %.2f \n \n' %
                   (epoch, train_loss, eval_loss, train_acc_rot, eval_acc_rot))
        f.write(text)

    # plot loss curve and accuracy curve into eps files
    fig_loss = plot_loss_fig(epoch, losses)
    fig_loss.savefig(os.path.join(result_path, 'fig_losses.eps'))
    plt.close(fig_loss)

    fig_acc = plot_acc_fig(epoch, accuracies[:, :2])
    fig_acc.savefig(os.path.join(result_path, 'fig_accuracies.eps'))
    plt.close(fig_acc)
# ========================================================== #
