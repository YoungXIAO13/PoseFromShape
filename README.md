# PoseFromShape
(BMVC 2019) PyTorch implementation of Paper "Pose from Shape: Deep Pose Estimation for Arbitrary 3D Objects"
[[PDF]](http://imagine.enpc.fr/~xiaoy/PoseFromShape/poseFromShape2019BMVC.pdf) [[Project webpage]](http://imagine.enpc.fr/~xiaoy/PoseFromShape/)

<p align="center">
<img src="https://github.com/YoungXIAO13/PoseFromShape/blob/master/img/teaser_test.png" width="400px" alt="teaser">
</p>

If our project is helpful for your research, please consider citing:
```Bash
@INPROCEEDINGS{Xiao2019PoseFromShape,
    author    = {Yang Xiao and Xuchong Qiu and Pierre{-}Alain Langlois and Mathieu Aubry and Renaud Marlet},
    title     = {Pose from Shape: Deep Pose Estimation for Arbitrary 3D Objects},
    booktitle = {BMVC},
    year      = {2019}}
```

## Table of Content
* [Installation](#installation)
* [Training](#training)
* [Testing](#testing)

## Installation

### Dependencies
The code can be used in **Linux** system with the the following dependencies: Python 3.6, Pytorch 1.0.1, Python-Blender 2.77, meshlabserver

We recommend to utilize conda environment to install all dependencies and test the code.

```shell
## Download the repository
git clone 'https://github.com/YoungXIAO13/PoseFromShape'
cd PoseFromShape

## Create python env with relevant packages
conda create --name PoseFromShape --file auxiliary/spec-file.txt
source activate PoseFromShape

## Install blender as a python module
conda install auxiliary/python-blender-2.77-py36_0.tar.bz2
```

### Datasets and Models
To download and prepare the datasets for training and testing (Pascal3D, ObjectNet3D, ShapeNetCore, SUN397, Pix3D, LineMod):
```shell
cd data
bash prepare_data.sh
```

To download the pretrained models (Pascal3D, ObjectNet3D, ShapeNetCore):
```shell
cd model
bash download_models.sh
```

## Training
To train on the Pascal3D dataset with real images:
```shell
cd run
bash train_Pascal3D.sh
```

To train on the ObjectNet3D dataset with real images:
```shell
cd run
bash train_ObjectNet3D.sh
```

To train on the ShapeNetCore dataset with synthetic images:
```shell
cd run
bash train_ShapeNetCore.sh
```

## Testing
While the network was trained on real or synthetic images, all the testing was done on real images.

### Pascal3D+
To test on the Pascal3D dataset with real images:
```shell
cd run
bash test_Pascal3D.sh
```

### ObjectNet3D
```shell
cd run
bash test_ObjectNet3D.sh
```
### Pix3D
```shell
cd run
bash test_Pix3D.sh
```

### LineMod
```shell
cd run
bash test_LineMod.sh
```

