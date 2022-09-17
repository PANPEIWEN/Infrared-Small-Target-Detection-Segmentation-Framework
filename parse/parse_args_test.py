# @Time    : 2022/4/7 21:38
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : parse_args_test.py
# @Software: PyCharm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Infrared Small Target Detection')
    parser.add_argument('--model', type=str, default='AGPCNet', help='select in the model folder')

    parser.add_argument('--dataset', type=str, default='NUAA',
                        help='NUAA, SIRST_AUG, SIRST_4M, IRSTD-1k')
    parser.add_argument('--checkpoint', type=str, default='NUAA_AGPCNet(Res34)_2022_09_14_19_26_47')
    parser.add_argument('--cur_best', type=str, default='best', help='current, best')
    parser.add_argument('--base_size', type=int, default=512, help='base image size')
    parser.add_argument('--crop_size', type=int, default=512, help='crop image size')
    parser.add_argument('--suffix', type=str, default='png')
    parser.add_argument('--data_aug', type=bool, default=False, help='data augmentation')
    parser.add_argument('--base_dir', type=str, default='/data1/ppw/works/All_ISTD/datasets',
                        help='your dataset directory')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N', help='dataloader threads')
    parser.add_argument('--criterion', type=str, default='SL', help='SL, BCE, CE')
    parser.add_argument('--gpus', type=str, default='6',
                        help='Select gpu to test')
    parser.add_argument('--ROC_thr', type=int, default=10, help='crop image size')

    args = parser.parse_args()
    return args
