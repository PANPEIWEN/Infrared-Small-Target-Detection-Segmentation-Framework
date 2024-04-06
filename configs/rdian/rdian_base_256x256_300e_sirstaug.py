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
        type='RDIAN'
    ),
    loss=dict(type='SoftIoULoss')
)

optimizer = dict(
    type='Adagrad',
    setting=dict(lr=0.005, weight_decay=1e-4)
)
runner = dict(type='EpochBasedRunner', max_epochs=300)
data = dict(
    train_batch=16,
    test_batch=16,
    rgb=False)