# Data preparation for PoseFromShape

Run ```bash prepare_data.sh``` to download all the datasets used in this project
and generate the **multi-view** renders for shape representation.

* This could take a very long time as there are 6 datasets to download 
and a large amount of images to generate.

* To accelerate the data preparation, you can run the bash file for one dataset
at a time.

## Point Cloud Generation

To generate **point cloud** from the .obj file for Pascal3D and ObjectNet3D.

* Fork the virtual scanner from [O-CNN](https://github.com/wang-ps/O-CNN/tree/master/virtual_scanner)
and build the executable file. This has been tested on ubuntu-16.04.

* Make sure that you have generated the correct obj files by running 
```python off2obj.py```
after downloading the dataset.

* To generate point cloud for Pascal3D, run:
```
python point_cloud.py --dataset_dir Pascal3D --dataset_format Pascal3D --input CAD --virtualscanner #Path_of_the_executable_virtual_scanner
```

* To generate point cloud for ObjectNet3D, run:
```
python point_cloud.py --dataset_dir ObjectNet3D --dataset_format Pascal3D --input CAD/obj --virtualscanner #Path_of_the_executable_virtual_scanner
```
