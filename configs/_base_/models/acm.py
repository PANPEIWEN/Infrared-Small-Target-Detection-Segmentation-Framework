model = dict(
    name='ACM',
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type=None,
        type_info='resnet'
    ),
    decode_head=dict(
        type='ASKCResNetFPN',
        layer_blocks=[4, 4, 4],
        channels=[8, 16, 32, 64],
        fuse_model='AsymBi'
    ),
    loss=dict(type='SoftIoULoss')
)
