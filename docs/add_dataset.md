## Add Custom Dataset

You need to follow the process below to add custom dataset.

### Dataset Preparation

Please refer
to [get_started.md](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework/blob/main/docs/get_started.md)
for dataset preparation.

### Add Dataset Config File

1. Create config file named _your_dataset_name.py_ in
   the [configs/\_base\_/datasets](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework/blob/main/configs/_base_/datasets)
   folder.
2. Config code specification:

```python
data = dict(
    # For identification, no practical use
    dataset_type='NUAA',
    # You dataset path
    data_root='/data1/ppw/works/All_ISTD/datasets/NUAA',
    # You want to resize the image size, this is the size of the image for training and testing if data_aug=False
    base_size=512,
    # You want to crop the image size, this is the size of the image for training and testing if data_aug=True
    crop_size=512,
    # Whether to use data augmentation, a variety of data augmentation will be added later for selection
    data_aug=True,
    # Suffix of the data image
    suffix='png',
    # DataLoader num_workers
    num_workers=8,
    # Train batch size
    train_batch=16,
    # Test batch size
    test_batch=8,
    # The filename where the training set is stored
    train_dir='trainval',
    # The filename where the testing set is stored
    test_dir='test'
)
```