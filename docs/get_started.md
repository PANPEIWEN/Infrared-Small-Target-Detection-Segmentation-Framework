## Installation
Our config mechanism is based on mmcv, so we need to install the mmcv series package.
```
# Python == 3.8
# Pytorch == 1.10
# Cuda == 11.1

conda create -n ABC python=3.8
conda install pytorch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia

pip install -U openmim
mim install mmcv-full==1.4.0
mim install mmdet==2.25.0
mim install mmsegmentation==0.28.0
```

After the installation is complete, if other packages are missing during the running process, you can install them directly with pip.
## Dataset Preparation

### File Structure
```
|-datasets
  |-NUAA
    |-trainval
      |-images
        |-Misc_1.png
        ...
      |-masks
        |-Misc_1.png
        ...
    |-test
      |-images
        |-Misc_1.png
        ...
      |-masks
        |-Misc_1.png
        ...
  |-IRSTD
  ...
```

### Datasets Link

https://drive.google.com/drive/folders/1RGpVHccGb8B4_spX_RZPEMW9pyeXOIaC?usp=sharing
