# @Time    : 2022/9/14 20:31
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : build_dataset.py
# @Software: PyCharm
from utils.data import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def build_dataset(dataset_name, base_size, crop_size, num_workers, train_batch, test_batch, local_rank):
    trainset = DatasetLoad(dataset_name, base_size, crop_size, 'train')
    testset = DatasetLoad(dataset_name, base_size, crop_size, 'test')
    if local_rank != -1:
        train_sample = DistributedSampler(trainset)
        train_data = DataLoader(dataset=trainset, batch_size=train_batch, sampler=train_sample,
                                pin_memory=True, num_workers=num_workers, drop_last=True)
        test_data = DataLoader(dataset=testset, batch_size=test_batch, shuffle=False,
                               num_workers=num_workers, drop_last=False)
    else:
        train_data = DataLoader(dataset=trainset, batch_size=train_batch, shuffle=True,
                                num_workers=num_workers, drop_last=True)
        test_data = DataLoader(dataset=testset, batch_size=test_batch, shuffle=False,
                               num_workers=num_workers, drop_last=False)
    return [train_sample, train_data, test_data, trainset.__len__(), testset.__len__()] if local_rank != -1 else [
        train_data, test_data, trainset.__len__(), testset.__len__()]
