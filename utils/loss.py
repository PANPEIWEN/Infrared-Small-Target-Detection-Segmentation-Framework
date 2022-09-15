# @Time    : 2022/4/6 14:58
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : loss.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLoULoss(nn.Module):
    def __init__(self):
        super(SoftLoULoss, self).__init__()

    def forward(self, pred, target):
        # Old One
        pred = torch.sigmoid(pred)
        smooth = 1

        # print("pred.shape: ", pred.shape)
        # print("target.shape: ", target.shape)

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)

        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.crit = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        target.squeeze(dim=1)
        loss = self.crit(pred, target)
        return loss

