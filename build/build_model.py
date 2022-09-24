# @Time    : 2022/9/14 20:10
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : build_model.py
# @Software: PyCharm
from mmcv import Config
from model.build_segmentor import Model


def build_model(cfg):
    model = Model(cfg)
    return model
