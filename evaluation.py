import numpy as np
from tqdm import tqdm
import torch
import os
from torch.utils.data import DataLoader

from auxiliary.utils import AverageValueMeter, get_pred_from_cls_output, rotation_acc, rotation_err


def val(data_loader, model, bin_size, shape, criterion_azi=None, criterion_ele=None, criterion_inp=None, criterion_reg=None):
    val_loss = AverageValueMeter()
    val_acc_rot = AverageValueMeter()
    predictions = torch.zeros([1, 3], dtype=torch.float).cuda()
    labels = torch.zeros([1, 3], dtype=torch.long).cuda()

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            # load data and label
            if shape is not None:
                im, shapes, label = data
                im, shapes, label = im.cuda(), shapes.cuda(), label.cuda()
            else:
                im, label = data
                im, label = im.cuda(), label.cuda()

            # forward pass
            out = model(im) if shape is None else model(im, shapes)

            # compute losses and update the meters
            if criterion_reg is not None:
                loss_azi = criterion_azi(out[0], label[:, 0])
                loss_ele = criterion_ele(out[1], label[:, 1])
                loss_inp = criterion_inp(out[2], label[:, 2])
                loss_reg = criterion_reg(out[3], out[4], out[5], label.float())
                loss = loss_azi + loss_ele + loss_inp + loss_reg
                val_loss.update(loss.item(), im.size(0))

            # transform the output into the label format
            preds = get_pred_from_cls_output([out[0], out[1], out[2]])
            for n in range(len(preds)):
                pred_delta = out[n + 3]
                delta_value = pred_delta[torch.arange(pred_delta.size(0)), preds[n].long()].tanh() / 2
                preds[n] = (preds[n].float() + delta_value + 0.5) * bin_size
            pred = torch.cat((preds[0].unsqueeze(1), preds[1].unsqueeze(1), preds[2].unsqueeze(1)), 1)

            # compute accuracy
            acc_rot = rotation_acc(pred, label.float())
            val_acc_rot.update(acc_rot.item(), im.size(0))

            # concatenate results and labels
            labels = torch.cat((labels, label), 0)
            predictions = torch.cat((predictions, pred), 0)

    predictions = predictions[1:, :]
    labels = labels[1:, :]

    return val_loss.avg, val_acc_rot.avg, predictions, labels


def test_category(shape, batch_size, dataset, dataset_test, model, bin_size, cat, predictions_path, logname):

    # initialize data loader and run validation
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    _, _, predictions, labels = val(test_loader, model, bin_size, shape)

    # save predictions
    out_file = os.path.join(predictions_path, 'predictions_{}.npy'.format(cat))
    np.save(out_file, predictions.cpu().numpy())

    # calculate the rotation errors between prediction and ground truth
    test_errs = rotation_err(predictions, labels.float()).cpu().numpy()
    Acc = 100. * np.mean(test_errs <= 30)
    Med = np.median(test_errs)

    with open(logname, 'a') as f:
        f.write('test accuracy for %d images of catgory %s in datatset %s \n' % (len(dataset_test), cat, dataset))
        f.write('Med_Err is %.2f, and Acc_pi/6 is %.2f \n \n' % (Med, Acc))

    return Acc, Med
