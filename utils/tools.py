# @Time    : 2022/9/14 22:11
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : tools.py
# @Software: PyCharm

import numpy as np
import torch
import random


def random_seed(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)


def sche_dict(args):
    args_dict = vars(args)
    sche = {}
    for k, v in args_dict.items():
        if k.startswith('sche_'):
            sche[k] = v
    return sche


def do_nothing():
    pass
