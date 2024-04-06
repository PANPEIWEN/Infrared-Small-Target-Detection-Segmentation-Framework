_base_ = [
    'agpcnet_res18_512x512_800e_nuaa.py',
]
# model settings
model = dict(
    decode_head=dict(
        backbone='resnet34')
)
data = dict(train_batch=4)
