# test on the novel categories using point cloud as input
python testing.py --dataset ObjectNet3D --shape PointCloud --model model/ObjectNet3D_novel_pointcloud.pth \
--output_dir ObjectNet3D_novel_pointcloud --shape_feature_dim 1024 --shape_dir pointcloud
