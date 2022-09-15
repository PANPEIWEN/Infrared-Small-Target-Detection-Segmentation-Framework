# @Time    : 2022/9/14 20:12
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : build_criterion.py
# @Software: PyCharm
import torch.nn as nn
from utils.loss import SoftLoULoss, CrossEntropy


def build_criterion(criterion_name):
    if criterion_name == 'SL':
        criterion = SoftLoULoss()
    elif criterion_name == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    elif criterion_name == 'CE':
        criterion = CrossEntropy()
    else:
        assert 'Criterion: ' + criterion_name + ' is not defined'
    return criterion
