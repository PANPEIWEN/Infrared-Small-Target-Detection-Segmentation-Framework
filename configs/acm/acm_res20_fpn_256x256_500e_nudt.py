_base_ = [
    '../_base_/datasets/nudt.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_500e.py',
    '../_base_/models/acm.py'
]
optimizer = dict(
    type='Adagrad',
    setting=dict(lr=0.05, weight_decay=1e-4)
)
runner = dict(type='EpochBasedRunner', max_epochs=1500)
