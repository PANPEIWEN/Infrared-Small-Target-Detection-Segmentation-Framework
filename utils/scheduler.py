# @Time    : 2022/5/31 17:19
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : scheduler.py
# @Software: PyCharm
import math


class PolyLR(object):
    def __init__(self, optimizer, num_epochs, base_lr, *args, **kwargs):
        super(PolyLR, self).__init__()
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.base_lr = base_lr
        for k, v in kwargs.items():
            self.power = v

    def step(self, epoch):
        lr = self.base_lr * (1 - epoch / self.num_epochs) ** self.power
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class CosineAnnealingLR(object):
    def __init__(self, optimizer, num_epochs, base_lr, *args, **kwargs):
        super(CosineAnnealingLR, self).__init__()
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.base_lr = base_lr
        for k, v in kwargs.items():
            self.min_lr = v

    def step(self, epoch):
        lr = self.min_lr + ((self.base_lr - self.min_lr) / 2) * (1 + math.cos(epoch / self.num_epochs * math.pi))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
