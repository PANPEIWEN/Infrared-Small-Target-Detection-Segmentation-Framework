# @Time    : 2022/4/6 20:22
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : save_model.py
# @Software: PyCharm
from datetime import datetime

import numpy as np
import os
import torch.nn as nn
import torch
from skimage import measure
import numpy


def make_dir(dataset, model):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = "%s_%s_%s" % (dataset, model, dt_string)
    os.makedirs('work_dirs/%s' % save_dir, exist_ok=True)
    return save_dir


def save_ckpt(state, save_path, filename):
    torch.save(state, os.path.join(save_path, filename))


def save_model_and_result(dt_string, epoch, train_loss, test_loss, best_iou, recall, precision, save_mIoU_dir,
                          save_other_metric_dir):
    with open(save_mIoU_dir, 'a') as f:
        f.write('{} - {:04d}:\t - train_loss: {:04f}:\t - test_loss: {:04f}:\t mIoU {:.4f}\n'.format(dt_string, epoch,
                                                                                                     train_loss,
                                                                                                     test_loss,
                                                                                                     best_iou))
    with open(save_other_metric_dir, 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epoch))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')
        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')
