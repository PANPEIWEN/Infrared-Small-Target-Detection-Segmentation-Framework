# Infrared-Small-Target-Segmentation-Framework

A general framework for infrared small target detection and segmentation. By modifying the configuration file, you can
adjust various parameters, switch models and datasets, and you can easily add your own models and datasets.

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

## Add your own model
