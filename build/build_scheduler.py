# @Time    : 2022/9/15 20:39
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : build_scheduler.py
# @Software: PyCharm
from utils.scheduler import *


def build_scheduler(optimizer, cfg):
    scheduler_name = cfg.lr_config['policy']
    scheduler_class = globals()[scheduler_name]
    scheduler = scheduler_class(optimizer=optimizer, base_lr=cfg.optimizer['setting']['lr'],
                                num_epochs=cfg.runner['max_epochs'], **cfg.lr_config)
    return scheduler
