# model settings
model = dict(
    name='DNANet',
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type=None,
        type_info='resnet',
    ),
    decode_head=dict(
        type='DNANet',
        num_classes=1,
        input_channels=3,
        block_name='resnet',
        num_blocks=[2, 2, 2, 2],
        nb_filter=[16, 32, 64, 128, 256]
    ),
    loss=dict(type='SoftIoULoss')
)
