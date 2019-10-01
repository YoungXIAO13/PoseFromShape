import os, sys
import numpy as np
import shutil
from tqdm import tqdm
from data_index import cat_id_to_desc, cat_desc_to_id, get_example_ids

sys.path.append('..')
from render_utils import render_obj_grid, render_obj_with_view


def render_example(example_id, render_dir, input_dir, output_dir, texture_dir, csv_file, shape, views):
    example_in_dir = os.path.join(input_dir, example_id)
    example_out_dir = os.path.join(output_dir, example_id)
    example_render_dir = os.path.join(render_dir, example_id)

    # Set obj file path
    try:
        obj = os.path.join(example_in_dir, 'models', 'model_normalized.obj')
    except:
        return False

    if os.path.isdir(example_out_dir):
        if len(os.listdir(example_out_dir)) == views:
            return False
        else:
            shutil.rmtree(example_out_dir)

    if not os.path.isdir(example_out_dir):
        os.makedirs(example_out_dir)
        
    if not os.path.isdir(example_render_dir):
        os.makedirs(example_render_dir)

    # Set texture images path
    if os.path.isdir(texture_dir):
        textures = [name for name in os.listdir(texture_dir)]
        textures.sort()
    else:
        raise ValueError('Invalid texture directory !')

    # redirect output to log file
    logfile = 'render.log'
    open(logfile, 'a').close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)

    textures = [name for name in os.listdir(texture_dir)]
    textures.sort()
    texture = textures[np.random.randint(0, len(textures))]
    texture_img = os.path.join(texture_dir, texture)
    
    # generate the synthetic training images and its multi-view reference images
    render_obj_grid(obj, example_render_dir, [512, 512], 30, 5, 1, 2, False, None, None)
    render_obj_with_view(obj, example_out_dir, csv_file, texture_img, views, shape)

    # disable output redirection
    os.close(1)
    os.dup(old)
    os.close(old)

    return True


def render_cat(input_dir, render_dir, output_dir, texture_dir, csv_file, cat_id, shape, views):
    example_ids = get_example_ids(input_dir, cat_id)
    if len(example_ids) > 200: example_ids = example_ids[:200]

    cat_in_dir = os.path.join(input_dir, cat_id)
    cat_out_dir = os.path.join(output_dir, cat_id)
    cat_render_dir = os.path.join(render_dir, cat_id)

    for example_id in tqdm(example_ids):
        render_example(example_id, cat_render_dir, cat_in_dir, cat_out_dir, texture_dir, csv_file, shape, views)


model_dir = 'ShapeNetCore.v2'
output_dir = 'Synthetic_training_images'
render_dir = 'Renders_semi_sphere'
texture_dir = 'textures'

cats = ['airplane', 'bag', 'bathtub', 'bed', 'birdhouse', 'bookshelf', 'bus', 'cabinet', 'camera', 'car', 
        'chair', 'clock', 'dishwasher', 'display', 'faucet', 'lamp', 'laptop', 'speaker', 'mailbox', 'microwave', 
        'motorcycle', 'piano', 'pistol', 'printer', 'rifle', 'sofa', 'table', 'train', 'watercraft', 'washer']
        
csv_file = 'synthetic_annotation.txt'
if not os.path.exists(csv_file):
    with open(csv_file, 'a') as f:
        f.write('image_path,cat_id,example_id,azimuth,elevation\n')

for cat in tqdm(cats):    
    cat_id = cat_desc_to_id(cat)
    render_cat(model_dir, render_dir, output_dir, texture_dir, csv_file, cat_id, [512, 512], 20)
    
os.system("rm render.log")
