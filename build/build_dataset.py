# @Time    : 2022/9/14 20:31
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : build_dataset.py
# @Software: PyCharm
from utils.data import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def build_dataset(dataset_name, base_size, crop_size, num_workers, train_batch, test_batch, local_rank):
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    if dataset_name == 'SIRST_AUG':
        trainset = SirstAugDataset(mode='train')
        testset = SirstAugDataset(mode='test')

    elif dataset_name == 'SIRST_4M':
        trainset = Sirst40000(mode='train', data_name=dataset_name)
        testset = Sirst40000(mode='test', data_name=dataset_name)
    elif dataset_name == 'NUAA':
        dataset_dir = 'datasets/' + '/' + dataset_name
        train_img_ids, val_img_ids, test_txt = load_dataset(
            'datasets/', dataset_name)
        trainset = TrainSetLoader(dataset_dir, img_id=train_img_ids, base_size=base_size,
                                  crop_size=crop_size,
                                  transform=input_transform, suffix='.png')
        testset = TestSetLoader(dataset_dir, img_id=val_img_ids, base_size=base_size, crop_size=crop_size,
                                transform=input_transform, suffix='.png')
    else:
        assert 'Dataset: ' + dataset_name + ' is not defined'
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
