# @Time    : 2022/5/31 17:19
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : scheduler.py
# @Software: PyCharm
import math


def linear_warmup(optimizer, epoch, base_lr, warmup_epoch=5):
    if epoch == 0:
        lr = base_lr / warmup_epoch
    else:
        lr = epoch * (base_lr / warmup_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class PolyLR(object):
    def __init__(self, optimizer, num_epochs, base_lr, warmup, **kwargs):
        super(PolyLR, self).__init__()
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.base_lr = base_lr
        self.warmup = warmup
        self.warmup_epoch = 5 if self.warmup else 0
        if 'sche_power' in kwargs:
            self.power = kwargs['sche_power']
        else:
            self.power = 1.0

    def step(self, epoch):
        if self.warmup and epoch <= self.warmup_epoch:
            linear_warmup(self.optimizer, epoch, self.base_lr, self.warmup_epoch)
        else:
            lr = self.base_lr * (1 - (epoch - self.warmup_epoch) / self.num_epochs) ** self.power
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr


class CosineAnnealingLR(object):
    def __init__(self, optimizer, num_epochs, base_lr, warmup, **kwargs):
        super(CosineAnnealingLR, self).__init__()
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.base_lr = base_lr
        self.warmup = warmup
        self.warmup_epoch = 5 if self.warmup else 0
        if 'sche_min_lr' in kwargs:
            self.min_lr = kwargs['sche_min_lr']
        else:
            self.min_lr = 1e-5

    def step(self, epoch):
        if self.warmup and epoch <= self.warmup_epoch:
            linear_warmup(self.optimizer, epoch, self.base_lr, self.warmup_epoch)
        else:
            lr = self.min_lr + ((self.base_lr - self.min_lr) / 2) * (
                        1 + math.cos((epoch - self.warmup_epoch) / self.num_epochs * math.pi))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
