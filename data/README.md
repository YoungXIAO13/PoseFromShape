# Data preparation for PoseFromShape

Run ```bash prepare_data.sh``` to download all the datasets used in this project
and generate the **multi-view** renders for shape representation.

* This could take a very long time as there are 6 datasets to download 
and a large amount of images to generate.

* To accelerate the data preparation, you can run the bash file for one dataset
at a time.

## Point Cloud Generation

To generate point clouds for Pascal3D and ObjectNet3D.

First make sure that you have generated the correct .obj files by running 
```
python off2obj.py
```

Then you need to get the dependencies for the virtual_canner:
```
apt-get install -y --no-install-recommends libboost-all-dev libcgal-dev libeigen3-dev
```
and build it:
```
cd virtual_scanner
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

You should find the **executable_virtual_scanner** in `./virtual_scanner/build/virtualscanner`

* To generate point cloud for Pascal3D, run:
```
python point_cloud.py --dataset_dir Pascal3D --dataset_format Pascal3D --input CAD --virtualscanner #Path_of_the_executable_virtual_scanner
```

* To generate point cloud for ObjectNet3D, run:
```
python point_cloud.py --dataset_dir ObjectNet3D --dataset_format Pascal3D --input CAD/obj --virtualscanner #Path_of_the_executable_virtual_scanner
```
