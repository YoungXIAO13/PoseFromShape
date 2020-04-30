#!/usr/bin/env bash
python inference.py --num_render=12 --shape='MultiView' \
--model='model/ObjectNet3D.pth' \
--image_path='demo/armadillo_images' \
--render_path='demo/armadillo_multiviews/crop' \
--obj_path='demo/armadillo.obj'
