import numpy as np
import json
import math
import os

data_list = json.load(open('pix3d.json'))

csv_file = 'Pix3D.txt'
if not os.path.exists(csv_file):
    with open(csv_file, 'a') as f:
        f.write('img_source,model_source,truncated,occluded,slightly_occluded,image_path,cat_id,example_id,model_name,azimuth,elevation,inplane_rotation\n')

for i in range(0, len(data_list)):
    img_source = data_list[i]['img_source']
    model_source = data_list[i]['model_source']
    truncated = data_list[i]['truncated']
    occluded = data_list[i]['occluded']
    slightly_occluded = data_list[i]['slightly_occluded']

    image_path = data_list[i]['img'].replace('img', 'crop')
    cat_id = data_list[i]['category']

    # retrieve example_id and model name from the model path
    model = data_list[i]['model']
    model_name = (os.path.split(model)[1])[:-4]
    example_id = os.path.split(os.path.split(model)[0])[1]

    # compute elevation and azimuth from cam_position
    cam_pose = data_list[i]['cam_position']
    elevation = math.degrees(math.atan(cam_pose[1] / math.sqrt(cam_pose[0]**2 + cam_pose[2]**2)))
    cam_pose[0] = -cam_pose[0]
    if cam_pose[2] == 0. or cam_pose[2] == -0.:
        if cam_pose[0] <= 0:
            azimuth = 90.
        else:
            azimuth = 270.
    else:
        azimuth = math.degrees(math.atan(cam_pose[0] / cam_pose[2]))
        if cam_pose[0] <= 0 and cam_pose[2] < 0:
            azimuth = azimuth
        elif cam_pose[0] > 0 and cam_pose[2] < 0:
            azimuth = 360. + azimuth
        else:
            azimuth = 180. + azimuth

    inplane_rotation = data_list[i]['inplane_rotation']

    with open(csv_file, 'a') as f:
        f.write(
            img_source + ',' + model_source + ',' + str(truncated) + ',' + str(occluded) + ',' + str(slightly_occluded)
            + ',' + image_path + ',' + cat_id + ',' + example_id + ',' + model_name + ',' +
            str(azimuth) + ',' + str(elevation) + ',' + str(inplane_rotation) + '\n')
