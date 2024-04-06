# @Time    : 2022/4/6 14:41
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : data.py
# @Software: PyCharm
import random
import sys
import os.path as osp
import os
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch
import numpy as np
import math


class DatasetLoad(Data.Dataset):
    def __init__(self, data_root, base_size, crop_size, mode, train_dir, test_dir, data_aug=True, suffix='png',
                 rgb=True, **kwargs):
        self.base_size = base_size
        self.crop_size = crop_size
        self.mode = mode
        self.data_aug = data_aug
        self.rgb = rgb
        assert mode in ['train', 'test'], 'The mode should be train or test'
        if mode == 'train':
            self.data_dir = osp.join(data_root, train_dir)
        else:
            self.data_dir = osp.join(data_root, test_dir)

        self.img_names = []
        for img in os.listdir(osp.join(self.data_dir, 'images')):
            if img.endswith(suffix):
                self.img_names.append(img)

        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        self.gray_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([-0.1246], [1.0923])
        ])

    def _sync_transform(self, img, mask):
        if self.mode == 'train' and self.data_aug:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            crop_size = self.crop_size
            long_size = random.randint(
                int(self.base_size * 0.5), int(self.base_size * 2.0))
            # int(self.base_size * 0.8), int(self.base_size * 1.2))
            w, h = img.size
            if h > w:
                oh = long_size
                ow = int(1.0 * w * long_size / h + 0.5)
                short_size = ow
            else:
                ow = long_size
                oh = int(1.0 * h * long_size / w + 0.5)
                short_size = oh
            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)
            if short_size < crop_size:
                padh = crop_size - oh if oh < crop_size else 0
                padw = crop_size - ow if ow < crop_size else 0
                img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
                mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            w, h = img.size
            x1 = random.randint(0, w - crop_size)
            y1 = random.randint(0, h - crop_size)
            img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
            mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
            if random.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
            img, mask = np.array(img), np.array(mask)
            img = self.rgb_transform(img) if self.rgb else self.gray_transform(img)
            mask = transforms.ToTensor()(mask)
        else:
            img = img.resize((self.base_size, self.base_size), Image.BILINEAR)
            mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)
            img, mask = np.array(img), np.array(mask)
            img = self.rgb_transform(img) if self.rgb else self.gray_transform(img)
            mask = transforms.ToTensor()(mask)
        return img, mask

    def __getitem__(self, item):
        img_name = self.img_names[item]
        img_path = osp.join(self.data_dir, 'images', img_name)
        label_path = osp.join(self.data_dir, 'masks', img_name)
        img = Image.open(img_path).convert('RGB') if self.rgb else Image.open(img_path).convert('L')
        mask = Image.open(label_path).convert('L')
        img, mask = self._sync_transform(img, mask)
        return img, mask

    def __len__(self):
        return len(self.img_names)
