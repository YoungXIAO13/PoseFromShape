import os
import scipy.io as sio
import numpy as np
from PIL import Image
from tqdm import trange
import cv2


cats = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']
databases = ['imagenet', 'pascal']
subsets = ['train', 'val']

annotation_file = 'Pascal3D.txt'
if not os.path.exists(annotation_file):
    with open(annotation_file, 'a') as f:
        f.write('source,set,cat,im_name,object,cad_index,truncated,occluded,difficult,azimuth,elevation,inplane_rotation,left,upper,right,lower,has_keypoints,distance,im_path\n')

for subset in subsets:
    for database in databases:
        if subset == 'val' and database == 'imagenet':
            continue
        
        for cat in cats:
            image_dir = os.path.join('Images', cat + '_' + database)
            annotation_dir = os.path.join('Annotations', cat + '_' + database)

            # read the validation image list
            if database == 'imagenet':
                val_list_file = os.path.join('Image_sets', '{}_imagenet_val.txt'.format(cat))
            else:
                val_list_file = os.path.join('PASCAL', 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', '{}_val.txt'.format(cat))
            with open(val_list_file, 'r') as f:
                val_list = [line.rstrip('\n') for line in f]

            # remove the classification label exiting in pascal image list
            if database == 'pascal':
                val_list = [str(line.split()[0]) for line in val_list]

            # read image names
            im_files = [name for name in os.listdir(image_dir)]
            im_files.sort()

            # read annotations and transform into csv files
            for n in trange(len(im_files), desc='reading category {} for {} set of {} database'.format(cat, subset, database)):
                im_file = im_files[n]
                im_name = str(im_file.split('.')[0])
                
                if subset == 'train' and database == 'pascal' and (im_name in val_list):
                    continue
                if subset == 'val' and not (im_name in val_list):
                    continue
            
                record = (sio.loadmat(os.path.join(annotation_dir, im_name + '.mat')))['record'][0][0]
                objects = record['objects'][0]
                for i in range(objects.shape[0]):
                    object_cls = objects[i]['class'].item()
                    if object_cls != cat:
                        continue
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
                    inplane_rotation = str(viewpoint['theta'].item())
                    if azimuth == '0' and elevation == '0' and inplane_rotation == '0':
                        continue
                    distance = str(viewpoint['distance'].item())
                    try:
                        anchors = objects[i]['anchors'][0][0]
                        has_keypoint = str(1) if len(anchors) > 0 else str(0)
                    except:
                        has_keypoint = str(0)
                    bbox = tuple((objects[i]['bbox'][0]).astype('int'))
                    left, upper, right, lower = str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])
                
                    with open(annotation_file, 'a') as f:
                        f.write(database + ',' + subset + ',' + cat + ',' + im_name + ',' + str(i) + ',' + cad_index + ',' +
                                truncated + ',' + occluded + ',' + difficult + ',' + azimuth + ',' + elevation + ',' + inplane_rotation + ',' + 
                                left + ',' + upper + ',' + right + ',' + lower + ',' + has_keypoint + ',' + distance + ',' + os.path.join(image_dir, im_file) + '\n')
