_base_ = [
    'acm_res20_fpn_512x512_800e_nuaa.py',
]
# model settings
model = dict(
    decode_head=dict(
        name='ASKCResUNet')
)
