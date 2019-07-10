import numpy as np
import torch
import shutil
import matplotlib.pyplot as plt
import os
from PIL import Image


def resize_padding(im, desired_size, mode="RGB"):
    # compute the new size
    old_size = im.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_im = Image.new(mode, (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_im


def KaiMingInit(net):
    """Kaiming Init layer parameters."""
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, a=0.2)  # slope = 0.2 in the original implementation
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)


def save_checkpoint(state, is_best, filename, result_path):
    """save state and best model ever"""
    torch.save(state, filename)
    if is_best:  # store model with best accuracy
        shutil.copyfile(filename, os.path.join(result_path, 'model_best.pth'))


def load_checkpoint(model, pth_file):
    """load state and network weights"""
    checkpoint = torch.load(pth_file, map_location=lambda storage, loc: storage.cuda())
    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Previous weight loaded')


class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.avg = self.avg * (self.count / (self.count + n)) + val * (n / (self.count + n))
        self.count += n


def get_pred_from_cls_output(outputs):
    preds = []
    for n in range(0, len(outputs)):
        output = outputs[n]
        _, pred = output.topk(1, 1, True, True)
        preds.append(pred.view(-1))
    return preds


def accuracy(outputs, targets):
    """Compute accuracy for each euler angle separately"""
    with torch.no_grad():  # no grad computation to reduce memory
        preds = get_pred_from_cls_output(outputs)
        res = []
        for n in range(0, len(outputs)):
            res.append(100. * torch.mean((preds[n] == targets[:, n]).float()))
        return res


def accuracyViews(outputs, targets, classes, views=(4, 8, 12)):
    """Compute accuracy for different number of views"""
    with torch.no_grad():  # no grad computation to reduce memory
        # compute the predictions
        _, preds = outputs.topk(1, 1)
        preds = preds.view(-1)
        # compute the accuracy according to number of views
        accs = []
        for view in views:
            preds_n = preds // (classes / view)
            targets_n = targets // (classes / view)
            correts_n = torch.eq(preds_n, targets_n)
            acc = correts_n.float().mean() * 100
            accs.append(acc)

        return accs


def gen_confusion(outputs, targets, azi_classes, ele_classes, rol_classes=None, epoch=0, result_dir="./"):
    """generate confusion matrix for phi and theta"""
    confusion_azi = torch.zeros(azi_classes, azi_classes)
    confusion_ele = torch.zeros(ele_classes, ele_classes)

    # split groundtruth for phi and theta
    target_azi = targets[:, 0]
    target_ele = targets[:, 1]

    # split output for phi and theta
    output_azi = outputs[:, 0:azi_classes]
    output_ele = outputs[:, azi_classes:azi_classes+ele_classes]

    _, pred_azi = output_azi.topk(1, 1, True, True)  # predicted class indices
    _, pred_ele = output_ele.topk(1, 1, True, True)

    if rol_classes is not None:
        confusion_rol = torch.zeros(rol_classes, rol_classes)
        target_rol = targets[:, 2]
        output_rol = outputs[:, azi_classes + ele_classes:]
        _, pred_rol = output_rol.topk(1, 1, True, True)

    # compute the confusion matrix
    for i in range(0, targets.size(0)):  # each row represents a ground-truth class
        confusion_azi[target_azi[i], pred_azi[i]] += 1
        confusion_ele[target_ele[i], pred_ele[i]] += 1
        if rol_classes is not None:
            confusion_rol[target_rol[i], pred_rol[i]] += 1

    # normalize the confusion matrix
    for i in range(0, azi_classes):
        confusion_azi[i] = confusion_azi[i] / confusion_azi[i].sum() if confusion_azi[i].sum() != 0 else 0
    for i in range(0, ele_classes):
        confusion_ele[i] = confusion_ele[i] / confusion_ele[i].sum() if confusion_ele[i].sum() != 0 else 0
    if rol_classes is not None:
        for i in range(0, rol_classes):
            confusion_rol[i] = confusion_rol[i] / confusion_rol[i].sum() if confusion_rol[i].sum() != 0 else 0

    # plot the confusion matrix and save it
    fig_conf_azi = plt.figure()
    ax = fig_conf_azi.add_subplot(111)
    cax = ax.matshow(confusion_azi.numpy(), vmin=0, vmax=1)
    fig_conf_azi.colorbar(cax)
    plt.xlabel('predicted class')
    plt.ylabel('actual class')
    plt.title('confusion matrix for azimuth')
    fig_name = 'fig_confusion_azi_' + str(epoch) + '.jpg'
    fig_conf_azi.savefig(os.path.join(result_dir, fig_name))
    plt.close(fig_conf_azi)

    fig_conf_ele = plt.figure()
    ax = fig_conf_ele.add_subplot(111)
    cax = ax.matshow(confusion_ele.numpy(), vmin=0, vmax=1)
    fig_conf_ele.colorbar(cax)
    plt.xlabel('predicted class')
    plt.ylabel('actual class')
    plt.title('confusion matrix for elevation')
    fig_name = 'fig_confusion_ele_' + str(epoch) + '.jpg'
    fig_conf_ele.savefig(os.path.join(result_dir, fig_name))
    plt.close(fig_conf_ele)
    if rol_classes is None:
        return False

    fig_conf_rol = plt.figure()
    ax = fig_conf_rol.add_subplot(111)
    cax = ax.matshow(confusion_rol.numpy(), vmin=0, vmax=1)
    fig_conf_rol.colorbar(cax)
    plt.xlabel('predicted class')
    plt.ylabel('actual class')
    plt.title('confusion matrix for inplane rotation')
    fig_name = 'fig_confusion_rol_' + str(epoch) + '.jpg'
    fig_conf_rol.savefig(os.path.join(result_dir, fig_name))
    plt.close(fig_conf_rol)
    return True


def plot_loss_fig(epoch, losses):
    epochs = np.arange(1, epoch + 2)
    fig_loss = plt.figure()
    plt.grid()
    if losses.shape[1] == 3:
        plt.plot(epochs, losses[0:epoch + 1, 0], 'b+-',
                 epochs, losses[0:epoch + 1, 1], 'g+-',
                 epochs, losses[0:epoch + 1, 2], 'r+-')
        plt.legend(('train_loss', 'val_loss', 'test_loss'), loc='upper right', fontsize='xx-small')
    else:
        plt.plot(epochs, losses[0:epoch + 1, 0], 'b+-',
                 epochs, losses[0:epoch + 1, 1], 'r+-')
        plt.legend(('train_loss', 'val_loss'), loc='upper right', fontsize='xx-small')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training curve')
    return fig_loss


def plot_acc_fig(epoch, accs):
    epochs = np.arange(1, epoch + 2)
    fig_acc = plt.figure()
    plt.grid()
    if accs.shape[1] == 3:
        plt.plot(epochs, accs[0:epoch + 1, 0], 'b+-',
                 epochs, accs[0:epoch + 1, 1], 'g+-',
                 epochs, accs[0:epoch + 1, 2], 'r+-')
        plt.legend(('train_acc', 'val_acc', 'test_acc'), loc='upper left', fontsize='xx-small')
    else:
        plt.plot(epochs, accs[0:epoch + 1, 0], 'b+-',
                 epochs, accs[0:epoch + 1, 1], 'r+-')
        plt.legend(('train_acc', 'val_acc'), loc='upper left', fontsize='xx-small')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy curve')
    return fig_acc


def plot_acc_angle_cls_fig(epoch, accuracies):
    epochs = np.arange(1, epoch + 2)
    fig_acc = plt.figure()
    plt.grid()
    plt.plot(epochs, accuracies[0:epoch + 1, 0], 'b+-',
             epochs, accuracies[0:epoch + 1, 1], 'bo--',
             epochs, accuracies[0:epoch + 1, 2], 'g+-',
             epochs, accuracies[0:epoch + 1, 3], 'go--',
             epochs, accuracies[0:epoch + 1, 4], 'r+-',
             epochs, accuracies[0:epoch + 1, 5], 'ro--')
    plt.legend(('train_azi', 'val_azi', 'train_ele', 'val_ele', 'train_rol', 'val_rol'), loc='upper left', fontsize='xx-small')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracies for euler angle classification')
    return fig_acc


def plot_angle_acc_fig(epoch, accs):
    epochs = np.arange(1, epoch + 2)
    fig_acc = plt.figure()
    plt.grid()
    if accs.shape[1] == 3:
        plt.plot(epochs, accs[0:epoch + 1, 0], 'r+-',
                 epochs, accs[0:epoch + 1, 1], 'g+-',
                 epochs, accs[0:epoch + 1, 2], 'b+-')
        plt.legend(('azimuth', 'elevation', 'inplane'), loc='upper left', fontsize='xx-small')
    else:
        plt.plot(epochs, accs[0:epoch + 1, 0], 'r+-',
                 epochs, accs[0:epoch + 1, 1], 'g+-')
        plt.legend(('azimuth', 'elevation'), loc='upper left', fontsize='xx-small')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy curve for angles')
    return fig_acc


def angles_to_matrix(angles):
    """Compute the rotation matrix from euler angles for a mini-batch"""
    azi = angles[:, 0]
    ele = angles[:, 1]
    rol = angles[:, 2]
    element1 = (torch.cos(rol) * torch.cos(azi) - torch.sin(rol) * torch.cos(ele) * torch.sin(azi)).unsqueeze(1)
    element2 = (torch.sin(rol) * torch.cos(azi) + torch.cos(rol) * torch.cos(ele) * torch.sin(azi)).unsqueeze(1)
    element3 = (torch.sin(ele) * torch.sin(azi)).unsqueeze(1)
    element4 = (-torch.cos(rol) * torch.sin(azi) - torch.sin(rol) * torch.cos(ele) * torch.cos(azi)).unsqueeze(1)
    element5 = (-torch.sin(rol) * torch.sin(azi) + torch.cos(rol) * torch.cos(ele) * torch.cos(azi)).unsqueeze(1)
    element6 = (torch.sin(ele) * torch.cos(azi)).unsqueeze(1)
    element7 = (torch.sin(rol) * torch.sin(ele)).unsqueeze(1)
    element8 = (-torch.cos(rol) * torch.sin(ele)).unsqueeze(1)
    element9 = (torch.cos(ele)).unsqueeze(1)
    return torch.cat((element1, element2, element3, element4, element5, element6, element7, element8, element9), dim=1)


def rotation_err(preds, targets):
    """compute rotation error for viewpoint estimation"""
    preds = preds.float().clone()
    targets = targets.float().clone()
    preds[:, 1] = preds[:, 1] - 180.
    preds[:, 2] = preds[:, 2] - 180.
    targets[:, 1] = targets[:, 1] - 180.
    targets[:, 2] = targets[:, 2] - 180.
    preds = preds * np.pi / 180.
    targets = targets * np.pi / 180.
    R_pred = angles_to_matrix(preds)
    R_gt = angles_to_matrix(targets)
    R_err = torch.acos(((torch.sum(R_pred * R_gt, 1)).clamp(-1., 3.) - 1.) / 2)
    R_err = R_err * 180. / np.pi
    return R_err


def rotation_acc(preds, targets, th=30.):
    R_err = rotation_err(preds, targets)
    return 100. * torch.mean((R_err <= th).float())


def angle_err(preds, targets):
    """compute rotation error for viewpoint estimation"""
    errs = torch.abs(preds - targets)
    errs = torch.min(errs, 360. - errs)
    return errs


if __name__ == '__main__':
    a = torch.randint(360, (4, 3)).float()
    b = torch.randint(360, (4, 3)).float()
    print(a)
    err = rotation_err(a, b)
    print(a)

