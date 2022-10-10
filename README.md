# Infrared-Small-Target-Segmentation-Framework
A general framework for infrared small target detection and segmentation. By modifying the configuration file, you can adjust various parameters, switch models and datasets, and you can easily add your own models and datasets.
## The tutorial and code are being improved...
## Dataset Preparation
### File Structure
&emsp;&emsp;----build  
&emsp;&emsp;......  
&emsp;&emsp;----datasets  
&emsp;&emsp;&emsp;----NUAA  
&emsp;&emsp;&emsp;&emsp;&emsp;----trainval  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;----images  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Misc_1.png  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;......  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;----masks  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Misc_1.png  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;......  
&emsp;&emsp;&emsp;&emsp;&emsp;----test  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;----images  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Misc_50.png  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;......  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;----masks  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Misc_50.png  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;......  
&emsp;&emsp;----model  
&emsp;&emsp;......
## Training
### Single GPU Training
```python train.py <CONFIG_FILE>```
### Multi GPU Training
```nproc_per_node``` is the number of gpus you are using.  

```python -m torch.distributed.launch --nproc_per_node=2 train.py <CONFIG_FILE>```
### Notes
* You can specify the GPU at the second line of ```os.environ['CUDA_VISIBLE_DEVICES']``` in train.py.
* Be sure to set args.local_rank to 0 if using Multi-GPU training.
## Test
```python test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE>```

