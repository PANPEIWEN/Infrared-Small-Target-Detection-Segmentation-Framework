# yapf:disable
log_config = dict(
    # During the training process, the log is saved every interval epoch.
    interval=10,
    # Identification role, no practical role
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='TensorboardLoggerHook')
    ])
# Distributed training backend
dist_params = dict(backend='nccl')

# Identification role, no practical role
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
