# train with multi view
python training.py --dataset ObjectNet3D --n_epoch 300 --lr 1e-4 --decrease 200 --random --keypoint \
--novel --shape MultiView --shape_dir Renders_semi_sphere --shape_feature_dim 2048

# train with point cloud
python training.py --dataset ObjectNet3D --n_epoch 300 --lr 1e-4 --decrease 200 --random --keypoint \
--novel --shape PointCloud --shape_dir pointcloud --shape_feature_dim 1024
