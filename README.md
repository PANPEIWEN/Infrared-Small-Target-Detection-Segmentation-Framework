# Infrared-Small-Target-Segmentation-Framework
A general framework for infrared small target detection and segmentation. By modifying the configuration file, you can adjust various parameters, switch models and datasets, and you can easily add your own models and datasets.
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
```python train.py```
### Multi GPU Training
```python -m torch.distributed.launch --nproc_per_node=2 train.py```
### Notes
* You can specify the GPU at the second line of ```os.environ['CUDA_VISIBLE_DEVICES']``` in train.py.
* Be sure to set args.local_rank to 0 if using Multi-GPU training.
## Test
```python test.py```
## Visualization
You can set parameters in parse_args_test.py.  
```python visualization.py```
## The tutorial and code are being improved...