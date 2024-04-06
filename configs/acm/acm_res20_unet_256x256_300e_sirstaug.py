_base_ = [
    'acm_res20_fpn_256x256_300e_sirstaug.py',
]
# model settings
model = dict(
    decode_head=dict(
        name='ASKCResUNet')
)
