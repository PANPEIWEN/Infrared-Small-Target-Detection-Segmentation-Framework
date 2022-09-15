# @Time    : 2022/4/6 21:16
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : test.py
# @Software: PyCharm
from tqdm import tqdm
import scipy.io as scio
from torchvision import transforms
from torch.utils.data import DataLoader
from build.build_model import build_model
from build.build_criterion import build_criterion
from build.build_dataset import build_dataset

from utils.data import *
from utils.metric import *
from utils.logs import *
from utils.save_model import *
from parse.parse_args_test import parse_args
import matplotlib.pyplot as plt


class Test(object):
    def __init__(self, args):
        super(Test, self).__init__()
        _, self.test_data, _, self.img_num = build_dataset(args.dataset, args.base_size, args.crop_size,
                                                           args.num_workers, 1, 1, -1)
        self.criterion = build_criterion(args.criterion)
        self.model = build_model(args.model)
        self.mIoU_metric = SigmoidMetric()
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)

        self.ROC = ROCMetric(1, 10)
        self.PD_FA = PD_FA(1, 10)
        self.best_recall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.best_precision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.mIoU_metric.reset()
        self.nIoU_metric.reset()

        self.device = torch.device('cuda:%s' % args.gpus if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load('work_dirs/' + args.checkpoint + '/' + args.cur_best + '.pth.tar')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = []
        with torch.no_grad():
            for i, (data, labels) in enumerate(tbar):
                data = data.to(self.device)
                labels = labels.to(self.device)
                if args.model == 'SINet':
                    _, _, _, pred = self.model(data)
                else:
                    pred, outs = self.model(data)
                '''feature map visualization'''
                for ii in range(16):
                    ax = plt.subplot(4, 4, ii + 1)
                    ax.axis('off')
                    plt.imshow(outs[ii // 4].data.cpu().numpy()[0, ii % 4, :, :], cmap='jet')
                plt.show()
                for ii in range(36):
                    ax = plt.subplot(6, 6, ii + 1)
                    ax.axis('off')
                    plt.imshow(outs[0].data.cpu().numpy()[0, ii, :, :], cmap='jet')
                plt.show()
                for ii in range(36):
                    ax = plt.subplot(6, 6, ii + 1)
                    ax.axis('off')
                    plt.imshow(outs[1].data.cpu().numpy()[0, ii, :, :], cmap='jet')
                plt.show()
                for ii in range(36):
                    ax = plt.subplot(6, 6, ii + 1)
                    ax.axis('off')
                    plt.imshow(outs[2].data.cpu().numpy()[0, ii, :, :], cmap='jet')
                plt.show()
                for ii in range(36):
                    ax = plt.subplot(6, 6, ii + 1)
                    ax.axis('off')
                    plt.imshow(outs[3].data.cpu().numpy()[0, ii, :, :], cmap='jet')
                plt.show()
                for ii in range(1):
                    ax = plt.subplot(1, 1, ii + 1)
                    ax.axis('off')
                    plt.imshow(pred.data.cpu().numpy()[0, ii, :, :], cmap='jet')
                plt.show()
                assert 1 == 2
                loss = self.criterion(pred, labels)
                losses.append(loss.item())
                self.ROC.update(pred, labels)
                self.mIoU_metric.update(pred, labels)
                self.nIoU_metric.update(pred, labels)
                self.PD_FA.update(pred, labels)
                _, mIoU = self.mIoU_metric.get()
                _, nIoU = self.nIoU_metric.get()
                ture_positive_rate, false_positive_rate, recall, precision, F1_score = self.ROC.get()
                tbar.set_description(
                    'Loss %.4f, mIoU %.4f, nIoU %.4f, F1-score %.4f' % (np.mean(losses), mIoU, nIoU, F1_score))
            FA, PD = self.PD_FA.get(self.img_num)
            save_result_for_test(args.checkpoint, args.model, args.epochs, mIoU, nIoU, recall, precision, FA, PD,
                                 args.dataset, F1_score)
            print('mIoU: %.4f, nIoU: %.4f, F1-score: %.4f' % (mIoU, nIoU, F1_score))


def main(args):
    tester = Test(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
