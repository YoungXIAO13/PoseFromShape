import torch.nn as nn
import torch
CE = nn.CrossEntropyLoss().cuda()
Huber = nn.SmoothL1Loss().cuda()


def cross_entropy_loss(pred, target, range):
    binSize = range // pred.size(1)
    trueLabel = target // binSize
    return CE(pred, trueLabel)


class CELoss(nn.Module):
    def __init__(self, range):
        super(CELoss, self).__init__()
        self.__range__ = range
        return

    def forward(self, pred, target):
        return cross_entropy_loss(pred, target, self.__range__)


def delta_loss(pred_azi, pred_ele, pred_rol, target, bin):
    # compute the ground truth delta value according to angle value and bin size
    target_delta = ((target % bin) / bin) - 0.5

    # compute the delta prediction in the ground truth bin
    target_label = (target // bin).long()
    delta_azi = pred_azi[torch.arange(pred_azi.size(0)), target_label[:, 0]].tanh() / 2
    delta_ele = pred_ele[torch.arange(pred_ele.size(0)), target_label[:, 1]].tanh() / 2
    delta_rol = pred_rol[torch.arange(pred_rol.size(0)), target_label[:, 2]].tanh() / 2
    pred_delta = torch.cat((delta_azi.unsqueeze(1), delta_ele.unsqueeze(1), delta_rol.unsqueeze(1)), 1)

    return Huber(5. * pred_delta, 5. * target_delta)


class DeltaLoss(nn.Module):
    def __init__(self, bin):
        super(DeltaLoss, self).__init__()
        self.__bin__ = bin
        return

    def forward(self, pred_azi, pred_ele, pred_rol, target):
        return delta_loss(pred_azi, pred_ele, pred_rol, target, self.__bin__)


def huber_loss(pred, target, bin):
    target_offset = 5. * (target % bin) / bin
    pred_offset = 5. * torch.sigmoid(pred)
    return Huber(pred_offset, target_offset)


class RegLoss(nn.Module):
    def __init__(self, bin):
        super(RegLoss, self).__init__()
        self.__bin__ = bin
        return

    def forward(self, pred, target):
        return huber_loss(pred, target, self.__bin__)


def geometry_cross_entropy_loss(pred, target, range):
    binNumber = pred.size(1)
    binSize = range / binNumber
    trueLabel = target // binSize
    leftLabel = ((trueLabel - 1) + binNumber) % binNumber
    rightLabel = (trueLabel + 1) % binNumber
    distance = (target - trueLabel * binSize).float()
    weight = distance / binSize
    if range == 360:
        loss = CE(pred, trueLabel) + weight * CE(pred, rightLabel) + (1. - weight) * CE(pred, leftLabel)
    else:
        notSmallest = (target != 0).float()
        notLargest = (target != (binNumber-1)).float()
        loss = CE(pred, trueLabel) + notLargest * weight * CE(pred, rightLabel) + notSmallest * (1. - weight) * CE(pred, leftLabel)
    return loss.mean()


class GeometryCELoss(nn.Module):
    def __init__(self, range):
        super(GeometryCELoss, self).__init__()
        self.__range__ = range
        return

    def forward(self, pred, target):
        return geometry_cross_entropy_loss(pred, target, self.__range__)


def exponential_geometry_cross_entropy_loss(pred, target, range, alpha):
    binNumber = pred.size(1)
    binSize = range / binNumber
    trueLabel = target // binSize
    leftLabel = ((trueLabel - 1) + binNumber) % binNumber
    rightLabel = (trueLabel + 1) % binNumber
    distance = (target - trueLabel * binSize).float()
    weight = torch.exp(-alpha * distance / binSize)
    if range == 360:
        loss = CE(pred, trueLabel) + weight * CE(pred, rightLabel) + (1. - weight) * CE(pred, leftLabel)
    else:
        notSmallest = (target != 0).float()
        notLargest = (target != (binNumber-1)).float()
        loss = CE(pred, trueLabel) + notLargest * weight * CE(pred, rightLabel) + notSmallest * (1. - weight) * CE(pred, leftLabel)
    return loss.mean()


class ExponentialGeometryCELoss(nn.Module):
    def __init__(self, range, alpha):
        super(ExponentialGeometryCELoss, self).__init__()
        self.__range__ = range
        self.__alpha__ = alpha
        return

    def forward(self, pred, target):
        return exponential_geometry_cross_entropy_loss(pred, target, self.__range__, self.__alpha__)


