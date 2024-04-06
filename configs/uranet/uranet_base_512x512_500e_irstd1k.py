_base_ = [
    '../_base_/datasets/irstd1k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_500e.py',
    '../_base_/models/uranet.py'
]

optimizer = dict(
    type='AdamW',
    setting=dict(lr=0.001, weight_decay=1e-4, betas=(0.9, 0.999))
)
data = dict(train_batch=8)
