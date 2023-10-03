_base_ = [
    '../_base_/datasets/nudt.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_500e.py',
    '../_base_/models/abcnet.py'
]
model = dict(
    decode_head=dict(
        dim=16,
    )
)

optimizer = dict(
    type='AdamW',
    setting=dict(lr=0.0001, weight_decay=0.01, betas=(0.9, 0.999))
)
data = dict(train_batch=32)
runner = dict(type='EpochBasedRunner', max_epochs=1500)
