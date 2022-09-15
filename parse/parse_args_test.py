# @Time    : 2022/4/7 21:38
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : parse_args_test.py
# @Software: PyCharm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Infrared Small Target Detection')
    parser.add_argument('--model', type=str, default='AGPCNet', help='select in the model folder')

    parser.add_argument('--dataset', type=str, default='SIRST_AUG',
                        help='NUAA, NUST, NUAA_AUG, SIRST_8000, SIRST_40000, SIRST_4M_T86')
    parser.add_argument('--checkpoint', type=str, default='NUAA_AGPCNet(Res34)_2022_09_14_19_26_47')
    parser.add_argument('--cur_best', type=str, default='best', help='current, best')
    parser.add_argument('--base_size', type=int, default=256, help='base image size')
    parser.add_argument('--crop_size', type=int, default=256, help='crop image size')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--split_method', type=str, default='idx_427', help='50_50, 10000_100')

    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (use to save log)')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')

    parser.add_argument('--ROC_thr', type=int, default=10, help='crop image size')

    args = parser.parse_args()
    return args
