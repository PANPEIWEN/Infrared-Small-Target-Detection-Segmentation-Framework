# @Time    : 2022/4/6 15:23
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : train.py
# @Software: PyCharm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'

from torch.nn import init
import torch.distributed

from utils.metric import *
from utils.logs import *
from utils.save_model import *
from utils.drawing import *
from utils.tools import *
from parse.parse_args_train import parse_args

from build.build_model import build_model
from build.build_criterion import build_criterion
from build.build_optimizer import build_optimizer
from build.build_dataset import build_dataset
from build.build_scheduler import build_scheduler


class Train(object):
    def __init__(self, args):
        super(Train, self).__init__()

        if args.local_rank != -1:
            device = torch.device('cuda', args.local_rank)
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            random_seed(42)
        else:
            self.device = torch.device(
                'cuda:%s' % args.gpus if torch.cuda.is_available() else 'cpu')
            random_seed(42)

        data = build_dataset(args.dataset, args.base_size, args.crop_size, args.num_workers, args.train_batch,
                             args.test_batch, args.local_rank, args.data_aug)
        if args.local_rank != -1:
            self.train_sample, self.train_data, self.test_data, self.train_data_len, self.test_data_len = data
        else:
            self.train_data, self.test_data, self.train_data_len, self.test_data_len = data

        self.model = build_model(args.model)

        if args.use_outer_init:
            self.model.apply(self.weight_init)
        if args.result_from:
            self.save_dir = args.checkpoint
            checkpoint = torch.load(
                'work_dirs/' + args.checkpoint + '/current.pth.tar')
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            if args.local_rank <= 0:
                self.save_dir = make_dir(args.dataset, args.model)
        if args.local_rank <= 0:
            save_train_args_log(args, self.save_dir)

        if args.local_rank != -1:
            self.model.to(device)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        else:
            self.model = self.model.to(self.device)
        print("Model Initializing")
        self.criterion = build_criterion(args.criterion)
        self.optimizer = build_optimizer(args.optimizer, self.model, args.lr)
        self.scheduler = build_scheduler(args.scheduler, self.optimizer, args.epochs, args.lr, args.warmup,
                                         **sche_dict(args))

        self.iou_metric = SigmoidMetric()
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        self.ROC = ROCMetric(1, 10)

        if args.result_from:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_mIoU = checkpoint['best_mIoU']
            self.best_nIoU = checkpoint['best_nIoU']
            self.best_f1 = checkpoint['best_f1']
            self.train_loss = checkpoint['train_loss']
            self.test_loss = checkpoint['test_loss']
            self.mIoU = checkpoint['mIoU']
            self.nIoU = checkpoint['nIoU']
            self.f1 = checkpoint['f1']
            self.num_epoch = checkpoint['num_epoch']
        else:
            self.best_mIoU = 0.0
            self.best_nIoU = 0.0
            self.best_f1 = 0.0
            self.train_loss = []
            self.test_loss = []
            self.mIoU = []
            self.nIoU = []
            self.f1 = []
            self.num_epoch = []

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.normal_(m.bias, 0)

    def training(self, epoch):
        self.model.train()
        losses = []
        # tbar = tqdm(self.train_data)
        if args.local_rank != -1:
            self.train_sample.set_epoch(epoch)
        self.scheduler.step(epoch - 1)
        for i, (img, mask) in enumerate(self.train_data):
            if args.local_rank != -1:
                img = img.cuda()
                mask = mask.cuda()
            else:
                img = img.to(self.device)
                mask = mask.to(self.device)
            pred = self.model(img)
            if args.model == 'SINet':
                loss_init = self.criterion(pred[0], mask) + self.criterion(pred[1], mask) + self.criterion(pred[2],
                                                                                                           mask)
                loss_final = self.criterion(pred[3], mask)
                loss = loss_final + loss_init
            else:
                loss = self.criterion(pred, mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            if args.local_rank <= 0:
                msg = 'Epoch %d/%d, Iter %d/%d, train loss %.4f, lr %.5f' % (
                    epoch, args.epochs, i + 1, self.train_data_len / args.train_batch / args.num_gpu,
                    np.mean(losses), self.optimizer.state_dict()['param_groups'][0]['lr'])
                print(msg)
                save_train_log(self.save_dir, epoch, args.epochs, i + 1,
                               self.train_data_len / args.train_batch / args.num_gpu,
                               np.mean(losses))
        if args.local_rank <= 0:
            save_ckpt({
                'epoch': epoch,
                'state_dict': self.model.module.state_dict() if args.local_rank != -1 else self.model.state_dict(),
                'loss': np.mean(losses),
                'optimizer': self.optimizer.state_dict(),
                'train_loss': self.train_loss,
                'test_loss': self.test_loss,
                'num_epoch': self.num_epoch,
                'best_mIoU': self.best_mIoU,
                'best_nIoU': self.best_nIoU,
                'best_f1': self.best_f1,
                'mIoU': self.mIoU,
                'nIoU': self.nIoU,
                'f1': self.f1
            }, save_path='work_dirs/' + self.save_dir, filename='current.pth.tar')
        self.train_loss.append(np.mean(losses))
        self.num_epoch.append(epoch)

    def testing(self, epoch):
        self.model.eval()
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        self.ROC.reset()
        eval_losses = []
        # tbar = tqdm(self.test_data)
        with torch.no_grad():
            for i, (img, mask) in enumerate(self.test_data):
                if args.local_rank != -1:
                    img = img.cuda()
                    mask = mask.cuda()
                else:
                    img = img.to(self.device)
                    mask = mask.to(self.device)
                if args.model == 'SINet':
                    _, _, _, pred = self.model(img)
                else:
                    pred = self.model(img)
                loss = self.criterion(pred, mask)
                eval_losses.append(loss.item())
                self.iou_metric.update(pred, mask)
                self.nIoU_metric.update(pred, mask)
                self.ROC.update(pred, mask)
                _, IoU = self.iou_metric.get()
                _, nIoU = self.nIoU_metric.get()
                _, _, _, _, F1_score = self.ROC.get()
                if args.local_rank <= 0:
                    msg = 'Epoch %d/%d, test loss %.4f, mIoU %.4f, nIoU %.4f, F1-score %.4f, ' \
                          'best_mIoU %.4f, ' \
                          'best_nIoU %.4f, best_F1-score %.4f' % (
                              epoch, args.epochs, np.mean(eval_losses), IoU, nIoU, F1_score, self.best_mIoU,
                              self.best_nIoU, self.best_f1)
            if args.local_rank <= 0:
                print(msg)
                save_test_log(self.save_dir, epoch, args.epochs, np.mean(eval_losses), IoU, nIoU, F1_score,
                              self.best_mIoU, self.best_nIoU, self.best_f1)
            self.test_loss.append(np.mean(eval_losses))
            self.mIoU.append(IoU)
            self.nIoU.append(nIoU)
            self.f1.append(F1_score)
            if args.local_rank <= 0:
                drawing_loss(self.num_epoch, self.train_loss, self.test_loss, self.save_dir)
                drawing_iou(self.num_epoch, self.mIoU, self.nIoU, self.save_dir)
                drawing_f1(self.num_epoch, self.f1, self.save_dir)
            if IoU > self.best_mIoU or nIoU > self.best_nIoU:
                if args.local_rank <= 0:
                    save_ckpt({
                        'epoch': epoch,
                        'state_dict': self.model.module.state_dict() if args.local_rank != -1 else self.model.state_dict(),
                        'loss': np.mean(eval_losses),
                        'mIoU': IoU,
                        'nIoU': nIoU,
                        'f1': F1_score
                    }, save_path='work_dirs/' + self.save_dir, filename='best.pth.tar')
            self.best_mIoU = max(IoU, self.best_mIoU)
            self.best_nIoU = max(nIoU, self.best_nIoU)
            self.best_f1 = max(F1_score, self.best_f1)


def main(args):
    trainer = Train(args)
    if args.local_rank != -1:
        torch.distributed.barrier()
    start = 1
    if args.result_from:
        checkpoint = torch.load(
            'work_dirs/' + args.checkpoint + '/current.pth.tar')
        start = checkpoint['epoch'] + 1
    for i in range(start, args.epochs + 1):
        trainer.training(i)
        trainer.testing(i)


if __name__ == '__main__':
    args = parse_args()
    main(args)
