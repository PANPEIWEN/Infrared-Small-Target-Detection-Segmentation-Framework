_base_ = [
    'dnanet_res10_512x512_800e_nuaa'
]
# model settings
model = dict(
    decode_head=dict(
        num_blocks=[2, 2, 2, 2]
    )
)