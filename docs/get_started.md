## Installation
Our config mechanism is based on mmcv, so we need to install the mmcv series package.
```
pip install -r requirements.txt
```
If you encounter the problem of mmcv-full installation failure, you can refer to https://github.com/open-mmlab/mmcv#installation

After the installation is complete, if other packages are missing during the running process, you can install them directly with pip.
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

### Datasets Link

https://drive.google.com/drive/folders/1RGpVHccGb8B4_spX_RZPEMW9pyeXOIaC?usp=sharing