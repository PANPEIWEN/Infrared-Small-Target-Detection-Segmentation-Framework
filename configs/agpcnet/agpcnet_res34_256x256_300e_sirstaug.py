_base_ = [
    '../_base_/datasets/sirstaug.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_500e.py',
    '../_base_/models/agpc.py'
]
model = dict(
    decode_head=dict(
        backbone='resnet34')
)

optimizer = dict(
    type='SGD',
    setting=dict(lr=0.05, momentum=0.9, weight_decay=0.0005)
)
runner = dict(type='EpochBasedRunner', max_epochs=300)
data = dict(train_batch=16)
