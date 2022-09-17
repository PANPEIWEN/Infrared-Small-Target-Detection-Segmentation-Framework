# @Time    : 2022/4/7 21:22
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : parse_args_train.py.py
# @Software: PyCharm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Infrared Small Target Detection')
    parser.add_argument('--model', type=str, default='AGPCNet', help='select in the model folder')

    parser.add_argument('--dataset', type=str, default='NUAA',
                        help='NUAA, SIRST_AUG, SIRST_4M, IRSTD-1k')
    parser.add_argument('--base_size', type=int, default=256, help='base image size')
    parser.add_argument('--crop_size', type=int, default=256, help='crop image size')
    parser.add_argument('--data_aug', type=bool, default=False,  help='data augmentation')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N', help='dataloader threads')

    parser.add_argument('--result_from', type=bool, default=False)
    parser.add_argument('--checkpoint', type=str, default='NUAA_AGPCNet(Res34)_2022_09_14_19_26_47')

    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--train_batch', type=int, default=32,
                        metavar='N', help='input batch size for training')
    parser.add_argument('--test_batch', type=int, default=8,
                        metavar='N', help='input batch size for testing')
    parser.add_argument('--criterion', type=str, default='SL', help='SL, BCE, CE')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='Adam, Adagrad and so on')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate')
    parser.add_argument('--warmup', type=bool, default=True)
    parser.add_argument('--scheduler', default='PolyLR',
                        choices=['CosineAnnealingLR', 'PolyLR'],
                        help='ExtraParam: CosineAnnealingLR: min_lr || PolyLR: power')
    parser.add_argument('--sche_power', default=1.0,
                        type=float, help='minimum learning rate')
    parser.add_argument('--use_outer_init', type=bool, default=False)
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example')
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    return args
