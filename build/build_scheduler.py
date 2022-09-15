# @Time    : 2022/9/15 20:39
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : build_scheduler.py
# @Software: PyCharm
from utils.scheduler import *


def build_scheduler(scheduler_name, optimizer, num_epochs, base_lr, ):
    if scheduler_name == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, num_epochs, base_lr, min_lr=1e-5)
    elif scheduler_name == 'PolyLR':
        scheduler = PolyLR(optimizer, num_epochs, base_lr, power=1.0)
    else:
        assert 'Scheduler: ' + scheduler_name + ' is not defined'
    return scheduler
