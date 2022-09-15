# @Time    : 2022/4/7 17:01
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : visual.py
# @Software: PyCharm
import os
import shutil

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from parse.parse_args_test import parse_args

args = parse_args()


def make_visualization_dir(target_image_path, target_dir):
    if not os.path.exists('work_dirs/' + args.checkpoint + '/' + 'visualization'):
        os.mkdir('work_dirs/' + args.checkpoint + '/' + 'visualization')

    if os.path.exists(target_image_path):
        shutil.rmtree(target_image_path)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_image_path)

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_dir)


def save_Pred_GT(pred, labels, target_image_path, dataset_name, num, suffix):
    img_name = os.listdir('datasets/' + dataset_name + '/test/images')
    val_img_ids = []
    for img in img_name:
        val_img_ids.append(img.split('.')[0])
    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(args.crop_size, args.crop_size))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) + suffix)
    img = Image.fromarray(labelsss.reshape(args.crop_size, args.crop_size))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)


def save_Pred_GT_visulize(pred, img_demo_dir, img_demo_index, suffix):
    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)

    img = Image.fromarray(predsss.reshape(args.crop_size, args.crop_size))
    img.save(img_demo_dir + '/' + '%s_Pred' % (img_demo_index) + suffix)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    img = plt.imread(img_demo_dir + '/' + img_demo_index + suffix)
    plt.imshow(img, cmap='gray')
    plt.xlabel("Raw Imamge", size=11)

    plt.subplot(1, 2, 2)
    img = plt.imread(img_demo_dir + '/' + '%s_Pred' % (img_demo_index) + suffix)
    plt.imshow(img, cmap='gray')
    plt.xlabel("Predicts", size=11)

    plt.savefig(img_demo_dir + '/' + img_demo_index + "_fuse" + suffix, facecolor='w', edgecolor='red')
    plt.show()


def total_visualization_generation(dataset_name, suffix, target_image_path, target_dir):
    source_image_path = 'datasets/' + dataset_name + '/test/images'
    ids = []
    img_name = os.listdir(source_image_path)
    for img in img_name:
        ids.append(img.split('.')[0])
    for i in range(len(ids)):
        source_image = source_image_path + '/' + ids[i] + suffix
        target_image = target_image_path + '/' + ids[i] + suffix
        shutil.copy(source_image, target_image)
    for i in range(len(ids)):
        source_image = target_image_path + '/' + ids[i] + suffix
        img = Image.open(source_image)
        img = img.resize((args.crop_size, args.crop_size), Image.ANTIALIAS)
        img.save(source_image)
    for m in range(len(ids)):
        print('Processing the %d image' % (m + 1))
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        img = plt.imread(target_image_path + '/' + ids[m] + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Raw Imamge", size=11)

        plt.subplot(1, 3, 2)
        img = plt.imread(target_image_path + '/' + ids[m] + '_GT' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Ground Truth", size=11)

        plt.subplot(1, 3, 3)
        img = plt.imread(target_image_path + '/' + ids[m] + '_Pred' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Predicts", size=11)
        plt.savefig(target_dir + '/' + ids[m].split('.')[0] + "_fuse" + suffix, facecolor='w', edgecolor='red')
