_base_ = [
    '../_base_/datasets/nudt.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_500e.py',
    '../_base_/models/agpc.py'
]

optimizer = dict(
    type='SGD',
    setting=dict(lr=0.05, momentum=0.9, weight_decay=0.0005)
)
runner = dict(type='EpochBasedRunner', max_epochs=1500)
