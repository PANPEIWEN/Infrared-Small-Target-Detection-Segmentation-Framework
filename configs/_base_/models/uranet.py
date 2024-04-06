# model settings
model = dict(
    name='UraNet_ffc',
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
        use_da=True,
        theta=0.7
    ),
    loss=dict(type='SoftIoULoss')
)
