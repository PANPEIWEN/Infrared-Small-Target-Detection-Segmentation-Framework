# @Time    : 2023/6/16 16:36
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : rebuild_train.py
# @Software: PyCharm
import argparse
import os
import time

import torch.distributed
import torch.nn
from mmcv import Config, DictAction

from utils.tools import *

from build.build_model import build_model
from build.build_criterion import build_criterion
from build.build_optimizer import build_optimizer
from build.build_dataset import build_dataset
from build.build_scheduler import build_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


class Train(object):
    def __init__(self, args, cfg):
        super(Train, self).__init__()
        self.cfg = cfg
        self.cfg.gpus = torch.cuda.device_count() if args.local_rank != -1 else 1
        self.resume = args.resume_from
        self.deep_supervision = 'deep_supervision' in self.cfg.model['decode_head']

        self.device = init_devices(args, self.cfg)

        data = build_dataset(args, self.cfg)
        self.data = init_data(args, data)

        model = build_model(self.cfg)
        self.model, checkpoint = init_model(args, self.cfg, model, self.device)
        self.criterion = build_criterion(self.cfg)
        optimizer = build_optimizer(self.model, self.cfg)
        if self.cfg.lr_config['policy']:
            self.scheduler = build_scheduler(optimizer, self.cfg)

        self.optimizer, self.metrics = init_metrics(args, optimizer, checkpoint if args.resume_from else None)
        self.save_dir, self.train_log_file, self.write = save_log(args, self.cfg, self.model)

    def training(self, epoch):
        self.model.train()
        losses = []
        if args.local_rank != -1:
            self.data['train_sample'].set_epoch(epoch)
        if not self.resume and self.cfg.lr_config['policy']:
            self.scheduler.step(epoch - 1)

        for i, data in enumerate(self.data['train_data']):
            since = time.time()
            img, mask = data2device(args, data, self.device)
            preds = self.model(img)
            loss, _ = compute_loss(preds, mask, self.deep_supervision, self.cfg, self.criterion)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            time_elapsed = time.time() - since
            show_log('train', args, self.cfg, epoch, losses, self.save_dir, self.train_log_file, i=i, data=self.data,time_elapsed=time_elapsed, optimizer=self.optimizer)
        save_model('train', args, self.cfg, epoch, self.model, losses, self.optimizer, self.metrics, self.save_dir, self.train_log_file)
        update_log('train', args, self.metrics, self.write, losses, epoch, optimizer=self.optimizer)

    def testing(self, epoch):
        self.model.eval()
        reset_metrics(self.metrics)
        eval_losses = []
        with torch.no_grad():
            for i, data in enumerate(self.data['test_data']):
                img, mask = data2device(args, data, self.device)
                preds = self.model(img)
                loss, preds = compute_loss(preds, mask, self.deep_supervision, self.cfg, self.criterion)
                eval_losses.append(loss.item())
                IoU, nIoU, F1_score = update_metrics(preds, mask, self.metrics)
            show_log('test', args, self.cfg, epoch, eval_losses, self.save_dir, self.train_log_file, IoU=IoU, nIoU=nIoU,F1_score=F1_score, metrics=self.metrics)
            append_metrics(args, self.metrics, eval_losses, IoU, nIoU, F1_score)
            save_model('test', args, self.cfg, epoch, self.model, eval_losses, self.optimizer, self.metrics,
                       self.save_dir, self.train_log_file, IoU=IoU, nIoU=nIoU)
            draw(args, self.metrics, self.save_dir, self.train_log_file)
            update_log('test', args, self.metrics, self.write, eval_losses, epoch, IoU=IoU, nIoU=nIoU,
                       F1_score=F1_score)


def main(args):
    cfg = Config.fromfile(args.config)
    trainer = Train(args, cfg)
    if args.local_rank != -1:
        torch.distributed.barrier()
    start = torch.load(args.resume_from)['epoch'] + 1 if args.resume_from else 1
    for i in range(start, cfg.runner['max_epochs'] + 1):
        trainer.training(i)
        trainer.testing(i)


if __name__ == '__main__':
    args = parse_args()
    main(args)
