# model settings
model = dict(
    name='AGPCNet',
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type=None,
        type_info='resnet',
    ),
    decode_head=dict(
        type='AGPCNet',
        backbone='resnet18',
        scalse=[10, 6, 5, 3],
        reduce_ratios=[16, 4],
        gca_type='patch',
        gca_att='post',
        drop=0.1),
    loss=dict(type='SoftIoULoss')
)
