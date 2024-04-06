_base_ = [
    '../_base_/datasets/irstd1k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_500e.py',
    '../_base_/models/acm.py'
]
optimizer = dict(
    type='Adagrad',
    setting=dict(lr=0.05, weight_decay=1e-4)
)
data = dict(train_batch=8)
