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
```shell
## Pascal3D+
cd Pascal3D
wget 'ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip'
unzip PASCAL3D+_release1.1.zip && rm PASCAL3D+_release1.1.zip
mv PASCAL3D+_release1.1/* ./ && rm -r PASCAL3D+_release1.1
rm CAD/*.mat && python off2obj.py
python create_annotation.py
python render_pascal3d.py

## ObjectNet3D
cd ..
wget 'ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_images.zip'
unzip ObjectNet3D_images.zip && rm ObjectNet3D_images.zip
cd ObjectNet3D
wget 'ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_cads.zip'
unzip ObjectNet3D_cads.zip && rm ObjectNet3D_cads.zip
mv ObjectNet3D/* ./ && rm -r ObjectNet3D
wget 'ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_annotations.zip'
unzip ObjectNet3D_annotations.zip && rm ObjectNet3D_annotations.zip
mv ObjectNet3D/* ./ && rm -r ObjectNet3D
wget 'ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_image_sets.zip'
unzip ObjectNet3D_image_sets.zip && rm ObjectNet3D_image_sets.zip
mv ObjectNet3D/* ./ && rm -r ObjectNet3D
python create_annotation.py
python render_object3d.py

cd ../
wget 'vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz'
python create_list.py
```

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
