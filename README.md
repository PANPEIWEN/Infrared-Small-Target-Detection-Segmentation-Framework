# The current code is too bloated, and we're refactoring the code, which will be released soon, and more models will be integrated. 🍺🍺🍺

# Infrared-Small-Target-Segmentation-Framework

A general framework for infrared small target detection and segmentation. By modifying or add the config file, you can
adjust various parameters, switch models and datasets and so on, and you can easily add your own models and datasets and so on.
It is recommended to spend a little time reading the [tutorial](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework#framework-usage-tutorial) before use, which can make you master the use of the framework faster.

## The tutorial and code are being improved...

## Installation

Please refer
to [get_started.md](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework/blob/main/docs/get_started.md)
for installation and dataset preparation.

## Training

### Single GPU Training

```
python train.py <CONFIG_FILE>
```

For example, train ACM model with fpn in single gpu, run:

```
python train.py configs/acm/acm_res20_fpn_512x512_800e_nuaa.py
```

### Multi GPU Training

```nproc_per_node``` is the number of gpus you are using.

```
python -m torch.distributed.launch --nproc_per_node=2 train.py <CONFIG_FILE>
```

For example, train ACM model with fpn and 2 gpus, run:

```
python -m torch.distributed.launch --nproc_per_node=2 train.py configs/acm/acm_res20_fpn_512x512_800e_nuaa.py
```

### Notes

* You can specify the GPU at the second line of ```os.environ['CUDA_VISIBLE_DEVICES']``` in train.py.
* Be sure to set args.local_rank to 0 if using Multi-GPU training.

## Test

```
python test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE>
```

For example, test ACM model with fpn, run:

```
python test.py configs/acm/acm_res20_fpn_512x512_800e_nuaa.py work_dirs/acm_res20_fpn_512x512_800e_nuaa/20221009_231431/best.pth.tar
```

If you want to visualize the result, you only add ```--show``` at the end of the above command.

## Framework usage tutorial
This part explains the config file in detail, which can make you understand the content of the config file more effectively and quickly, and master the overall framework.
### Add custom model
Please refer
to [add_model.md](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework/blob/main/docs/add_model.md)
for add custom model.
### Add custom dataset
Please refer
to [add_dataset.md](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework/blob/main/docs/add_dataset.md)
for add custom dataset.
### Add loss function
Please refer
to [add_loss.md](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework/blob/main/docs/add_loss.md)
for add loss function.

_Notice: Although you need to use the loss function already in pytorch, you still need to do this._
### Add optimizer and scheduler
Please refer
to [add_optimizer.md](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework/blob/main/docs/add_optimizer.md)
for add optimizer and scheduler.

_Notice: Although you need to use the optimizer and scheduler already in pytorch, you still need to do this._
