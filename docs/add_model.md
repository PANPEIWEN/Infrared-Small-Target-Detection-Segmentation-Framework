## Add your own model

You need to follow the process below to add your own model.

### Add Model File

1. Create a new **Python Package** named _YourModel_ in
   the [model](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework/blob/main/model) folder.
2. Create model file named _yourmodel.py_ in the **YourModel** folder.
3. Model code specification:

```python
class YourModelName(nn.Module):
    """
    1. You must add **kwargs.
    2. If you want to use deep_supervision, you can add deep_supervision.
    Note: That parameter names can only be 'deep_supervision'.
          If you use deep_supervision, the final output must be at the end 
          of the output list, for example:
          >>> out1 = self.conv1(x)
          >>> out2 = self.conv2(x)
          >>> out = self.final_conv(x)
          >>> return [out1, out2, out] if self.deep_supervision else out
       
    """

    def __init__(self, args1, args2, deep_supervision=True, **kwargs):
        super(YourModelName, self).__init__()
        pass

    def forward(self, x):
        pass

```

4. Modify [build_segmentor.py](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework/blob/main/model/build_segmentor.py)
file:

```python
# add
from model.YourModel.yourmodel import YourModelName

__all__ = [..., 'YourModelName']
```

### Add Model Config File

1. Create a new directory named _yourmodel_ in
   the [configs](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework/blob/main/configs) folder.
2. Create config file named _yourmodel_base_512x512_800e_nuaa.py_ in the **yourmodel** folder.
   _Tips: Recommended config file name naming rules:_
   ```[model_name]_[model_scale]_[data_size]_[train_epoch]_[dataset_name]```
3. Config code specification:

```python
"""
In the base is the config you inherited, you can modify the places you need to
modify after inheritance. For example, you want to modify train and test batch,
you can write like this:
    >>> data = dict(
    >>> train_batch=32,
    >>> test_batch=32)
You can use this method flexibly to make the config file more concise.
"""
_base_ = [
    # dataset config file
    '../_base_/datasets/nuaa.py',
    # run config file
    '../_base_/default_runtime.py',
    # optimizer and schedule config file
    '../_base_/schedules/schedule_500e.py'
]

# model settings
model = dict(
    # YourModelName
    name='YourModelName',
    type='EncoderDecoder',

    # If your model has a separate backbone, that:
    #  >>> type=ClassName
    # type_info only represent information, no practical use.
    # This code cannot be deleted.
    backbone=dict(
        type=None,
        type_info='resnet',
    ),

    # The type must be the same as your model class name.
    # The parameters are the parameters inside your model __init__.
    decode_head=dict(
        type='YourModelName',
        args1=...,
        args2=...,
        deep_supervision=True,
        ...
    ),

    # The type must in build_criterion.py with __all__.
    # If you want to set some parameters, you just need to add a key-value pair after type.
    # For example:
    # >>> loss=dict(type='BCEWithLogits', reduction='mean')
    loss=dict(type='SoftIoULoss')
)

# The type must in torch.optim, you need set parameters in setting.
# There cannot be key-value pairs that are not in the optimizer parameter list here.
optimizer = dict(
    type='SGD',
    setting=dict(lr=0.01, momentum=0.9, weight_decay=0.0005)
)
```