# PoseFromShape
(BMVC 2019)PyTorch implementation of Paper "Pose from Shape: Deep Pose Estimation for Arbitrary 3D Objects"


## Table of Content

## Installation

### Dependencies
The code can be used in **Linux** system with the the following dependencies: Python 3.6, Pytorch 1.0.1, Python-Blender 2.77

We recommend to utilize conda environment to install all dependencies and test the code.

```shell
## Download the repository
git clone ...
cd PoseFromShape

## Create python env with relevant packages
conda create --name PoseFromShape --file auxiliary/spec-file.txt
source activate PoseFromShape

## Install blender as a pytho module
conda install auxiliary/python-blender-2.77-py36_0.tar.bz2
```

### Datasets
Download and prepare the datasets for training and testing


### Models
To download pretrained models (Pascal3D, ObjectNet3D, ShapeNetSyn):
```shell
cd model
bash download_model.sh
```

## Training


## Testing

### Pascal3D+

### ObjectNet3D

### Pix3D

### LineMod

## Demo
