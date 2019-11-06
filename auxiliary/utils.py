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


def angles_to_matrix(angles):
    """Compute the rotation matrix from euler angles for a mini-batch.
    This is a PyTorch implementation computed by myself for calculating
    R = Rz(inp) Rx(ele - pi/2) Rz(-azi)
    
    For the original numpy implementation in StarMap, you can refer to:
    https://github.com/xingyizhou/StarMap/blob/26223a6c766eab3c22cddae87c375150f84f804d/tools/EvalCls.py#L20
    """
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
    
    # get elevation and inplane-rotation in the right format
    # R = Rz(inp) Rx(ele - pi/2) Rz(-azi)
    preds[:, 1] = preds[:, 1] - 180.
    preds[:, 2] = preds[:, 2] - 180.
    targets[:, 1] = targets[:, 1] - 180.
    targets[:, 2] = targets[:, 2] - 180.
    
    # change degrees to radians
    preds = preds * np.pi / 180.
    targets = targets * np.pi / 180.
    
    # get rotation matrix from euler angles
    R_pred = angles_to_matrix(preds)
    R_gt = angles_to_matrix(targets)
    
    # compute the angle distance between rotation matrix in degrees
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

