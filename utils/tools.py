# @Time    : 2022/9/14 22:11
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : tools.py
# @Software: PyCharm

import random
import torch.distributed
import torch.nn
from utils.metric import *
from torch.utils.tensorboard import SummaryWriter
from utils.logs import *
import shutil
from utils.save_model import *
from utils.drawing import *
import logging


def random_seed(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)


def empty_function():
    pass


def model_wrapper(model_dict):
    new_dict = {}
    for k, v in model_dict.items():
        new_dict['decode_head.' + k] = v
    return new_dict


def init_metrics(args, optimizer, checkpoint=None):
    best_mIoU, best_nIoU, best_f1 = 0.0, 0.0, 0.0
    train_loss, test_loss, mIoU, nIoU, f1, num_epoch = [], [], [], [], [], []
    if args.resume_from:
        best_mIoU = checkpoint['best_mIoU']
        best_nIoU = checkpoint['best_nIoU']
        best_f1 = checkpoint['best_f1']
        train_loss = checkpoint['train_loss']
        test_loss = checkpoint['test_loss']
        mIoU = checkpoint['mIoU']
        nIoU = checkpoint['nIoU']
        f1 = checkpoint['f1']
        num_epoch = checkpoint['num_epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
    iou_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    ROC = ROCMetric(1, 10)

    return optimizer, {'best_mIoU': best_mIoU, 'best_nIoU': best_nIoU, 'best_f1': best_f1, 'train_loss': train_loss,
                       'test_loss': test_loss, 'mIoU': mIoU, 'nIoU': nIoU, 'f1': f1, 'num_epoch': num_epoch,
                       'iou_metric': iou_metric, 'nIoU_metric': nIoU_metric, 'ROC': ROC}


def init_data(args, data):
    train_sample = None
    if args.local_rank != -1:
        train_sample, train_data, test_data, train_data_len, test_data_len = data
    else:
        train_data, test_data, train_data_len, test_data_len = data
    return {'train_sample': train_sample, 'train_data': train_data, 'test_data': test_data,
            'train_data_len': train_data_len, 'test_data_len': test_data_len}


def init_model(args, cfg, model, device):
    checkpoint = None
    if args.load_from:
        cfg.load_from = args.load_from
        checkpoint = torch.load(args.load_from)
        model.load_state_dict(checkpoint)

    # FIXME Loss Accuracy Decreases When Use resume_from
    if args.resume_from:
        cfg.resume_from = args.resume_from
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint['state_dict'])
    print("Model Initializing")

    if args.local_rank != -1:
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        model = model.to(device)

    return model, checkpoint


def init_devices(args, cfg):
    if args.local_rank != -1:
        device = torch.device('cuda', args.local_rank)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend=cfg.dist_params['backend'])
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    random_seed(42)
    return device


def save_log(args, cfg, model):
    save_dir, train_log_file_name, write = None, None, None
    if args.local_rank <= 0:
        save_dir = args.config.split('/')[-1][:-3]
        train_log_file_name = train_log_file()
        make_log_dir(save_dir, train_log_file_name)
        save_config_log(cfg, save_dir, train_log_file_name)
        save_model_struct(save_dir, train_log_file_name, model)
        if 'develop' in cfg:
            shutil.copy(cfg.develop['source_file_root'],
                        os.path.join('work_dirs', save_dir, train_log_file_name, 'model.py'))
        write = SummaryWriter(log_dir='work_dirs/' + save_dir + '/' + train_log_file_name + '/tf_logs')
    return save_dir, train_log_file_name, write


def data2device(args, data, device):
    img, mask = data
    if args.local_rank != -1:
        img = img.cuda()
        mask = mask.cuda()
    else:
        img = img.to(device)
        mask = mask.to(device)
    return img, mask


def compute_loss(preds, mask, deep_supervision, cfg, criterion):
    # TODO when use deep supervision, should log pred loss, not all loss sum
    if deep_supervision and cfg.model['decode_head']['deep_supervision']:
        loss = []
        for pre in preds:
            loss.append(criterion(pre, mask))
        loss = sum(loss)
        preds = preds[-1]
    else:
        loss = criterion(preds, mask)
    return loss, preds


def show_log(mode, args, cfg, epoch, losses, save_dir, train_log_file, **kwargs):
    if mode not in ['train', 'test']:
        raise ValueError('The parameter "mode" input should be "train" or "test"')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%F %T')
    if args.local_rank <= 0:
        if mode == 'train':
            msg = 'Epoch %d/%d, Iter %d/%d, train loss %.4f, lr %.5f, time %.5f' % (
                epoch, cfg.runner['max_epochs'], kwargs['i'] + 1,
                kwargs['data']['train_data_len'] / cfg.data['train_batch'] / cfg.gpus,
                np.mean(losses), kwargs['optimizer'].state_dict()['param_groups'][0]['lr'], kwargs['time_elapsed'])
            logging.info(msg)
            if (kwargs['i'] + 1) % cfg.log_config['interval'] == 0:
                save_train_log(save_dir, train_log_file, epoch, cfg.runner['max_epochs'], kwargs['i'] + 1,
                               kwargs['data']['train_data_len'] / cfg.data['train_batch'] / cfg.gpus,
                               np.mean(losses), kwargs['optimizer'].state_dict()['param_groups'][0]['lr'],
                               kwargs['time_elapsed'])
        else:
            msg = 'Epoch %d/%d, test loss %.4f, mIoU %.4f, nIoU %.4f, F1-score %.4f, best_mIoU %.4f, ' \
                  'best_nIoU %.4f, best_F1-score %.4f' % (
                      epoch, cfg.runner['max_epochs'], np.mean(losses), kwargs['IoU'], kwargs['nIoU'],
                      kwargs['F1_score'], kwargs['metrics']['best_mIoU'], kwargs['metrics']['best_nIoU'],
                      kwargs['metrics']['best_f1'])
            logging.info(msg)
            save_test_log(save_dir, train_log_file, epoch, cfg.runner['max_epochs'],
                          np.mean(losses), kwargs['IoU'], kwargs['nIoU'], kwargs['F1_score'],
                          kwargs['metrics']['best_mIoU'], kwargs['metrics']['best_nIoU'], kwargs['metrics']['best_f1'])


def save_model(mode, args, cfg, epoch, model, losses, optimizer, metrics, save_dir, train_log_file, **kwargs):
    if mode not in ['train', 'test']:
        raise ValueError('The parameter "mode" input should be "train" or "test"')
    if args.local_rank <= 0:
        ckpt_info = {
            'epoch': epoch,
            'state_dict': model.module.state_dict() if args.local_rank != -1 else model.state_dict(),
            'loss': np.mean(losses),
            'optimizer': optimizer.state_dict(),
            'train_loss': metrics['train_loss'],
            'test_loss': metrics['test_loss'],
            'num_epoch': metrics['num_epoch'],
            'best_mIoU': metrics['best_mIoU'],
            'best_nIoU': metrics['best_nIoU'],
            'best_f1': metrics['best_f1'],
            'mIoU': metrics['mIoU'],
            'nIoU': metrics['nIoU'],
            'f1': metrics['f1']
        }
        if mode == 'train':
            save_ckpt(ckpt_info, save_path='work_dirs/' + save_dir + '/' + train_log_file, filename='last.pth.tar')
            if cfg.checkpoint_config['by_epoch'] and epoch % cfg.checkpoint_config['interval'] == 0:
                save_ckpt(ckpt_info, save_path='work_dirs/' + save_dir + '/' + train_log_file,
                          filename='epoch_%d' % epoch + '.pth.tar')
        else:
            if kwargs['IoU'] > metrics['best_mIoU'] or kwargs['nIoU'] > metrics['best_nIoU']:
                save_ckpt(ckpt_info, save_path='work_dirs/' + save_dir + '/' + train_log_file, filename='best.pth.tar')
            if kwargs['IoU'] > metrics['best_mIoU']:
                save_ckpt(ckpt_info, save_path='work_dirs/' + save_dir + '/' + train_log_file,
                          filename='best_mIoU.pth.tar')
            if kwargs['nIoU'] > metrics['best_nIoU']:
                save_ckpt(ckpt_info, save_path='work_dirs/' + save_dir + '/' + train_log_file,
                          filename='best_nIoU.pth.tar')


def update_log(mode, args, metrics, write, losses, epoch, **kwargs):
    if mode not in ['train', 'test']:
        raise ValueError('The parameter "mode" input should be "train" or "test"')
    if args.local_rank <= 0:
        if mode == 'train':
            metrics['train_loss'].append(np.mean(losses))
            metrics['num_epoch'].append(epoch)
            write.add_scalar('train/train_loss', np.mean(losses), epoch)
            write.add_scalar('train/lr', kwargs['optimizer'].state_dict()['param_groups'][0]['lr'], epoch)
        else:
            metrics['best_mIoU'] = max(kwargs['IoU'], metrics['best_mIoU'])
            metrics['best_nIoU'] = max(kwargs['nIoU'], metrics['best_nIoU'])
            metrics['best_f1'] = max(kwargs['F1_score'], metrics['best_f1'])
            write.add_scalar('train/test_loss', np.mean(losses), epoch)
            write.add_scalar('test/mIoU', kwargs['IoU'], epoch)
            write.add_scalar('test/nIoU', kwargs['nIoU'], epoch)
            write.add_scalar('test/F1-score', kwargs['F1_score'], epoch)


def reset_metrics(metrics):
    metrics['iou_metric'].reset()
    metrics['nIoU_metric'].reset()
    metrics['ROC'].reset()


def update_metrics(preds, mask, metrics):
    metrics['iou_metric'].update(preds, mask)
    metrics['nIoU_metric'].update(preds, mask)
    metrics['ROC'].update(preds, mask)
    _, IoU = metrics['iou_metric'].get()
    _, nIoU = metrics['nIoU_metric'].get()
    _, _, _, _, F1_score = metrics['ROC'].get()
    return IoU, nIoU, F1_score


def append_metrics(args, metrics, losses, IoU, nIoU, F1_score):
    if args.local_rank <= 0:
        metrics['test_loss'].append(np.mean(losses))
        metrics['mIoU'].append(IoU)
        metrics['nIoU'].append(nIoU)
        metrics['f1'].append(F1_score)


def draw(args, metrics, save_dir, train_log_file):
    if args.local_rank <= 0:
        drawing_loss(metrics['num_epoch'], metrics['train_loss'], metrics['test_loss'], save_dir, train_log_file)
        drawing_iou(metrics['num_epoch'], metrics['mIoU'], metrics['nIoU'], save_dir, train_log_file)
        drawing_f1(metrics['num_epoch'], metrics['f1'], save_dir, train_log_file)

