_base_ = [
    'acm_res20_fpn_512x512_500e_irstd1k.py',
]
# model settings
model = dict(
    decode_head=dict(
        name='ASKCResUNet')
)
