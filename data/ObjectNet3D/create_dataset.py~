import os
import scipy.io as sio
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2


train_list = open('Image_sets/train.txt', 'r').read().split('\n')[:-1]
val_list = open('Image_sets/val.txt', 'r').read().split('\n')[:-1]
test_list = open('Image_sets/test.txt', 'r').read().split('\n')[:-1]

annotation_file = 'ObjectNet3D.txt'
if not os.path.exists(annotation_file):
    with open(annotation_file, 'a') as f:
        f.write('source,set,cat,im_name,object,cad_index,truncated,occluded,difficult,azimuth,elevation,inplane_rotation,left,upper,right,lower,has_keypoints,distance,im_path\n')

def read_annotation(im_name, image_dir, mat_dir, image_set):
    im_file = '{}.JPEG'.format(im_name)
    #im = cv2.imread(im_file)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #im = Image.fromarray(im).copy()
    try:
        record = (sio.loadmat(os.path.join(mat_dir, '{}.mat'.format(im_name))))['record'][0][0]
    except:
        print('MISS DATA', im_name)
        return False
    objects = record['objects'][0]
    for i in range(objects.shape[0]):
        object_cls = objects[i]['class'].item()
        truncated = str(objects[i]['truncated'].item())
        occluded = str(objects[i]['occluded'].item())
        difficult = str(objects[i]['difficult'].item())
        cad_index = str(objects[i]['cad_index'].item())
        viewpoint = objects[i]['viewpoint'][0][0]
        try:
            azimuth = str(viewpoint['azimuth'].item())
            elevation = str(viewpoint['elevation'].item())
        except:
            azimuth = str(viewpoint['azimuth_coarse'].item())
            elevation = str(viewpoint['elevation_coarse'].item())
        distance = str(viewpoint['distance'].item())
        inplane_rotation = str(viewpoint['theta'].item())
        #if azimuth == '0' and elevation == '0' and inplane_rotation == '0':
        #    continue
        bbox = tuple((objects[i]['bbox'][0]).astype('int'))
        left, upper, right, lower = str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])        
        try:
            anchors = objects[i]['anchors'][0]
            has_keypoint = str(1)
        except:
            has_keypoint = str(0)
        with open(annotation_file, 'a') as f:
            f.write('object3d' + ',' + image_set + ',' + object_cls + ',' + im_name + ',' + str(i) + ',' + cad_index + ',' +
                    truncated + ',' + occluded + ',' + difficult + ',' + azimuth + ',' + elevation + ',' + inplane_rotation + ',' + 
                    left + ',' + upper + ',' + right + ',' + lower + ',' + has_keypoint + ',' + distance + ',' + os.path.join(image_dir, im_file) + '\n')    


for im in tqdm(train_list):
    read_annotation(im, 'Images', 'Annotations', 'train')

for im in tqdm(val_list):
    read_annotation(im, 'Images', 'Annotations', 'val')

for im in tqdm(test_list):
    read_annotation(im, 'Images', 'Annotations', 'test')
