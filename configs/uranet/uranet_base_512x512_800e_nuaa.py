_base_ = [
    '../_base_/datasets/nuaa.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_500e.py'
]
# model settings
model = dict(
    name='URANet',
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type=None,
        type_info='resnet',
    ),
    decode_head=dict(
        type='URANet',
        in_channel=3,
        base_dim=32,
        class_num=1,
        bilinear=True,
        use_da=True
    ),
    loss=dict(type='SoftIoULoss')
)
optimizer = dict(
    type='SGD',
    setting=dict(lr=0.05, momentum=0.9, weight_decay=0.0005)
)
