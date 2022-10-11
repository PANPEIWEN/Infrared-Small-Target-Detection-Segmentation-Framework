## Add Optimizer and Scheduler

You need to follow the process below to add optimizer and scheduler.

### Add Optimizer

1. Optimizer does not need to be rewritten like loss function, you only need to change the value of type to \_\_all__
   of [build/build_optimizer.py](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework/blob/main/build/build_optimizer.py)
   , and then
   pass in the corresponding parameters in the setting of optimizer. The details will be described next.

### Add Scheduler

In order to easily combine warmup and scheduler, all our schedulers do not use pytorch, but rewrite them by themselves,
and do not inherit the loss function in pytorch like some loss functions.

You need to perform these operations
in [utils/scheduler.py](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework/blob/main/utils/scheduler.py)
.

1. Add scheduler

```python
class YourScheduler(object):
    # Where optimizer, base_lr, warmup and warmup_epochs is a necessary parameter, and the default values of warmup and warmup_epochs are None and 0 respectively.
    # num_epochs represents the total number of epochs for training, not every scheduler requires this parameter, but if it is required, the parameter name must be num_epochs. 
    def __init__(self, optimizer, base_lr, num_epochs, args1, args2, ..., warmup=None, warmup_epochs=0, **kwargs):
        super(YourScheduler, self).__init__()
        # The next four lines should be written in this format
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup = warmup
        self.warmup_epoch = warmup_epochs if self.warmup else 0
        pass

    def step(self, epoch):
        # The learning rate policy needs to follow the format
        if self.warmup and epoch <= self.warmup_epoch:
            # warmup
            globals()[self.warmup](self.optimizer, self.args1, self.args2, ...)
            # For example:
            # >>> globals()[self.warmup](self.optimizer, epoch, self.base_lr, self.warmup_epoch)
        else:
            # Calculate the learning rate
            lr = ...
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
```

2. Add the scheduler class name to \_\_all__ in
   the [build/build_scheduler.py](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework/blob/main/build/build_scheduler.py)
   .

### Add Warmup

You need to perform these operations
in [utils/scheduler.py](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework/blob/main/utils/scheduler.py)
.

1. Add warmup

```python
def your_warmup(optimizer, args1, args2, ...):
    ...
    # Calculate the learning rate
    lr = ...
    # Update learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

2. Add the scheduler function name to \_\_all__ in
   the [build/build_scheduler.py](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework/blob/main/build/build_scheduler.py)
   .

### Modify Config File

The settings of optimizer, scheduler and warmup are concentrated in
the [configs/\_base_/schedules/schedule_500e.py](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework/blob/main/configs/_base_/schedules/schedule_500e.py)
file, which has other settings. Next, we
will introduce the configuration file in detail.

```python
"""
Since no method has been found to rewrite the optimizer in pytorch, it is recommended to rewrite the optimizer dictionary in the final config file to cover it, which is only for illustration here.
Please refer to docs/add_model.md for details.
"""
optimizer = dict(
    # The type must in __all__ of build/build_optimizer.py.
    type='SGD',
    # Set the parameters of the optimizer, since there is no **kwargs parameter, the parameters set here can only be parameters common to all optimizers.
    # So it is recommended to rewrite the optimizer dictionary in the final configuration file to overwrite it.
    setting=dict(lr=0.01)
)

# No practical use
optimizer_config = dict()

"""
Choose your scheduler and warmup strategy, the policy and warmup must in __all__ of build/build_scheduler.py, the first letter is capitalized policy, the first letter is lowercase warmup.
The parameters required by scheduler and warmup can be passed in directly by adding key-value pairs.
"""
lr_config = dict(policy='PolyLR', warmup='linear', power=0.9, min_lr=1e-4, warmup_epochs=5)

# Number of training epochs
runner = dict(type='EpochBasedRunner', max_epochs=500)
# If by_epoch=True, the checkpoint is saved every interval epoch.
checkpoint_config = dict(by_epoch=True, interval=1)
# It has no practical effect at present, and this function will be implemented in the future.
# Validate every epochval epoch.
evaluation = dict(epochval=1)

```