_base_ = [
    'acm_res20_fpn_256x256_500e_nudt.py',
]
# model settings
model = dict(
    decode_head=dict(
        name='ASKCResUNet')
)
