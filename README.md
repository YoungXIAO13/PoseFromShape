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
To train on the ObjectNet3D dataset with real images and coarse alignment:
```shell
cd run
bash train_ObjectNet3D.sh
```

To train on the Pascal3D dataset with real images and coarse alignment:
```shell
cd run
bash train_Pascal3D.sh
```

To train on the ShapeNetCore dataset with synthetic images and precise alignment:
```shell
cd run
bash train_ShapeNetCore.sh
```

## Testing
While the network was trained on real or synthetic images, all the testing was done on real images.

### ObjectNet3D
```shell
cd run
bash test_ObjectNet3D.sh
```
You should obtain the results in Table 1 in the paper (*indicates testing on the **novel categories**):

| Method | bed | bookcase | calculator | cellphone | computer | door | cabinet | guitar | iron | knife | microwave | pen | pot | rifle | shoe | slipper | stove | toilet | tub | wheelchair | Average |
| :------: |:------: | :------: | :------: | :------: | :------: |:------: | :------: |:------: | :------: |:------: | :------: |:------: | :------: |:------: | :------: |:------: | :------: |:------: | :------: |:------: | :------: |
| [StarMap](https://arxiv.org/pdf/1803.09331.pdf) | 73 | 78 | 91 | 57 | 82 | - | 84 | 73 | 3 | 18 | 94 | 13 | 56 | 4 | - | 12 | 87 | 71 | 51 | 60 | 56 |
| [StarMap](https://arxiv.org/pdf/1803.09331.pdf)* | 37 | 69 | 19 | 52 | 73 | - | 78 | 61 | 2 | 9 | 88 | 12 | 51 | 0 | - | 11 | 82 | 41 | 49 | 14 | 42 |
| Ours(MV) | **82** | **90** | **95** | **65** | **93** | **97** | **89** | **75** | **52** | **32** | **95** | **54** | **82** | **45** | **67** | **46** | **95** | **82** | 67 | **66** | **73** |
| Ours(MV)* | 65 | **90** | 88 | **65** | 84 | 93 | 84 | 67 | 2 | 29 | 94 | 47 | 79 | 15 | 54 | 32 | 89 | 61 | **68** | 39 | 62 |



### Pascal3D+
To test on the Pascal3D dataset with real images:
```shell
cd run
bash test_Pascal3D.sh
```
You should obtain the results in Table 2 in the paper (*indicates **category-agnostic**):

| Method | Accuracy | Median Error |
| :------: |:------: | :------: |
| [Keypoints and Viewpoints](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Tulsiani_Viewpoints_and_Keypoints_2015_CVPR_paper.pdf) | 80.75 | 13.6 |
| [Render for CNN](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Su_Render_for_CNN_ICCV_2015_paper.pdf) | 82.00 | 11.7 |
| [Mousavian](http://openaccess.thecvf.com/content_cvpr_2017/papers/Mousavian_3D_Bounding_Box_CVPR_2017_paper.pdf) | 81.03 | 11.1 |
| [Grabner](https://zpascal.net/cvpr2018/Grabner_3D_Pose_Estimation_CVPR_2018_paper.pdf) | **83.92** | 10.9 |
| [Grabner](https://zpascal.net/cvpr2018/Grabner_3D_Pose_Estimation_CVPR_2018_paper.pdf)* | 81.33 | 11.5 |
| [StarMap](https://arxiv.org/pdf/1803.09331.pdf)* | 81.67 | 12.8 |
| Ours(MV)* | 82.66 | **10.0** |

### Pix3D
```shell
cd run
bash test_Pix3D.sh
```
You should obtain the results in Table 3 in the paper (*Accuracy* / *MedErr*):

| Method | Bed | Chair | Desk |
| :------: |:------: | :------: | :------: |
| [Georgakis](https://arxiv.org/pdf/1811.07249.pdf) | 50.8 / 28.6 | 31.2 / 57.3 | 34.9 / 51.6 |
| Ours(MV) | **59.8 / 20.0** | **52.4 / 26.6** | **56.6 / 26.6** |
