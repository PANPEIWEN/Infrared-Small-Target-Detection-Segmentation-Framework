_base_ = [
    '../_base_/datasets/sirstaug.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_500e.py'
]

model = dict(
    name='Segformer',
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type=None,
    ),
    decode_head=dict(
        type='UIUNet',
        deep_supervision=True
    ),
    loss=dict(type='SoftIoULoss')
)

optimizer = dict(
    type='Adam',
    setting=dict(lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
)
runner = dict(type='EpochBasedRunner', max_epochs=300)
data = dict(
    train_batch=8,
    test_batch=8)
