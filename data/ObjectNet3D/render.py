import os, sys
import bpy
import math
from math import radians
from tqdm import tqdm

sys.path.append('..')
from render_utils import resize_padding, makeLamp, parent_obj_to_camera, clean_obj_lamp_and_mesh, render_obj_grid

    
model_dir = 'CAD/obj'
render_dir = 'Renders_semi_sphere'

cats = [name for name in os.listdir(model_dir)]
cats.sort()
for cat in tqdm(cats):
    cat_models = [name for name in os.listdir(os.path.join(model_dir, cat))]
    cat_models.sort()
    for model in tqdm(cat_models):
        obj = os.path.join(model_dir, cat, model)
        out = os.path.join(render_dir, cat, model.split(".")[0])
        if not os.path.isdir(out):
            os.makedirs(out)
        
        # redirect output to log file
        logfile = 'render.log'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)
            
        # render object reference images
        render_obj_grid(obj, out, [512, 512], 30, 5, 1, 1.5, False, '-Y', 'Z')
        
        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)
        
os.system("rm render.log")
