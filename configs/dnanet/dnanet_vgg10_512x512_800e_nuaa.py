_base_ = [
    'dnanet_res10_512x512_800e_nuaa.py'
]
# model settings
model = dict(
    backbone=dict(
        type=None,
        type_info='vgg',
    ),
    decode_head=dict(
        block='vgg',
        num_blocks=[1, 1, 1, 1]
    )
)