# @Time    : 2022/4/7 15:30
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : visualization.py
# @Software: PyCharm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from tqdm import tqdm
from build.build_model import build_model
from build.build_dataset import build_dataset
from utils.save_model import *
from utils.visual import *
from parse.parse_args_test import parse_args


class Visualization(object):
    def __init__(self, args):
        super(Visualization, self).__init__()
        _, self.test_data, _, _ = build_dataset(args.dataset, args.base_size, args.crop_size,
                                                args.num_workers, 1, 1, -1)
        self.device = torch.device('cuda:%s' % args.gpus if torch.cuda.is_available() else 'cpu')
        self.model = build_model(args.model)
        checkpoint = torch.load('work_dirs/' + args.checkpoint + '/best.pth.tar')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        visualization_path = 'work_dirs/' + args.checkpoint + '/' + 'visualization' + '/' + 'visualization_result'
        visualization_fuse_path = 'work_dirs/' + args.checkpoint + '/' + 'visualization' + '/' + 'visualization_fuse'
        make_visualization_dir(visualization_path, visualization_fuse_path)

        tbar = tqdm(self.test_data)
        with torch.no_grad():
            num = 0
            for i, (data, labels) in enumerate(tbar):
                data = data.to(self.device)
                labels = labels.to(self.device)
                pred = self.model(data)
                save_Pred_GT(pred, labels, visualization_path, args.dataset, num, '.png')
                num += 1
            total_visualization_generation(args.dataset, '.png', visualization_path, visualization_fuse_path)
            print('Finishing')


def main(args):
    visual = Visualization(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
