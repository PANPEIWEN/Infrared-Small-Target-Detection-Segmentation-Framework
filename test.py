# @Time    : 2023/10/1 20:01
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : rebuild_test.py
# @Software: PyCharm
import argparse
import os
from mmcv import Config
from tqdm import tqdm
from build.build_model import build_model
from build.build_criterion import build_criterion
from build.build_dataset import build_dataset

from utils.visual import *
from utils.tools import *


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as txt'))
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed testing)')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


class Test(object):
    def __init__(self, args, cfg):
        super(Test, self).__init__()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%F %T')
        self.cfg = cfg
        self.deep_supervision = 'deep_supervision' in self.cfg.model['decode_head']
        self.save_dir = args.work_dir if args.work_dir else os.path.dirname(os.path.abspath(args.checkpoint))
        self.show_dir = args.show_dir if args.show_dir else os.path.join(self.save_dir, 'show')
        make_show_dir(self.show_dir) if args.show else empty_function()
        _, self.test_data, _, self.img_num = build_dataset(args, self.cfg)
        self.criterion = build_criterion(self.cfg)
        self.model = build_model(self.cfg)
        self.mIoU_metric = SigmoidMetric()
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        self.ROC = ROCMetric(1, 10)
        self.PD_FA = PD_FA(1, 10, cfg)
        self.best_recall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.best_precision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.mIoU_metric.reset()
        self.nIoU_metric.reset()
        self.PD_FA.reset()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(args.checkpoint)
        self.model.load_state_dict(checkpoint['state_dict'])
        logging.info("Model Initializing")
        self.model = self.model.to(self.device)
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = []

        with torch.no_grad():
            for i, (img, mask) in enumerate(tbar):
                img, mask = data2device(args, (img, mask), self.device)
                preds = self.model(img)
                loss, preds = compute_loss(preds, mask, self.deep_supervision, cfg, self.criterion)
                losses.append(loss.item())

                self.ROC.update(preds, mask)
                self.mIoU_metric.update(preds, mask)
                self.nIoU_metric.update(preds, mask)
                self.PD_FA.update(preds, mask)
                _, mIoU = self.mIoU_metric.get()
                _, nIoU = self.nIoU_metric.get()
                ture_positive_rate, false_positive_rate, recall, precision, F1_score = self.ROC.get()
                tbar.set_description(
                    'Loss %.4f, mIoU %.4f, nIoU %.4f, F1-score %.4f' % (np.mean(losses), mIoU, nIoU, F1_score))
                if args.show:
                    save_Pred_GT(preds, mask, self.show_dir, cfg.data['test_batch'] * i, cfg)
            FA, PD = self.PD_FA.get(self.img_num)
            save_test_config(cfg, self.save_dir)
            save_result_for_test(self.save_dir, mIoU, nIoU, recall, precision, FA, PD, F1_score, ture_positive_rate,
                                 false_positive_rate)
            if args.show:
                total_show_generation(self.show_dir, cfg)
                logging.info('Finishing')
            logging.info('mIoU: %.4f, nIoU: %.4f, F1-score: %.4f' % (mIoU, nIoU, F1_score))


def main(args):
    cfg = Config.fromfile(args.config)
    tester = Test(args, cfg)


if __name__ == "__main__":
    args = parse_args()
    main(args)
