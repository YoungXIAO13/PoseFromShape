import os, sys
import pandas as pd
from tqdm import tqdm
import numpy as np

sys.path.append('..')
from render_utils import resize_padding, makeLamp, parent_obj_to_camera, clean_obj_lamp_and_mesh, render_obj_grid


input_dir = 'model'
output_dir = 'Renders_semi_sphere'
csv_file = 'Pix3D.txt'

df = pd.read_csv(csv_file)
cats = np.unique(df.cat_id)
for cat in tqdm(cats):
    df_cat = df[df.cat_id == cat]
    examples = np.unique(df_cat.example_id)
    for example in tqdm(examples):
        df_example = df_cat[df_cat.example_id == example]
        model_names = np.unique(df_example.model_name)
        for model_name in (model_names):
            # set input obj file path
            obj = os.path.join(input_dir, cat, example, '{}.obj'.format(model_name))
                
            # create output dir
            if model_name == 'model':
                example_out_dir = os.path.join(output_dir, cat, example)
            else:
                example_out_dir = os.path.join(output_dir, cat, example, model_name)
            if not os.path.isdir(example_out_dir):
                os.makedirs(example_out_dir)
                
            # redirect output to log file
            logfile = 'render.log'
            open(logfile, 'a').close()
            old = os.dup(1)
            sys.stdout.flush()
            os.close(1)
            os.open(logfile, os.O_WRONLY)

            # Render object without texture
            render_obj_grid(obj, example_out_dir, [512, 512], 30, 5, 1, 1.5, False, None, None)

            # disable output redirection
            os.close(1)
            os.dup(old)
            os.close(old)

os.system("rm render.log")
