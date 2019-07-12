import os
import cv2
from tqdm import tqdm
import numpy as np


root_dir = 'SUN397'

cats_sim = [name for name in os.listdir(root_dir)]
cats_sim.sort()

for cat_sim in tqdm(cats_sim):
    if not os.path.isdir(os.path.join(root_dir, cat_sim)):
        continue
    cats = [name for name in os.listdir(os.path.join(root_dir, cat_sim))]
    cats.sort()

    for cat in cats:
        cat_dir = os.path.join(root_dir, cat_sim, cat)
        cat_out = os.path.join(cat_sim, cat)

        subdirs = [name for name in os.listdir(cat_dir)]
        if os.path.isdir(os.path.join(cat_dir, subdirs[0])):
            for subdir in subdirs:
                subdir_path = os.path.join(cat_dir, subdir)
                subdir_out = os.path.join(cat_out, subdir)
                if not os.path.isdir(subdir_out):
                    os.makedirs(subdir_out)
                imgs = [name for name in os.listdir(subdir_path)]
                for img in imgs:
                    im = cv2.imread(os.path.join(subdir_path, img))
                    if im is None:
                        im_resize = np.ones((256, 256, 3), dtype=np.uint8) * 255
                    else:
                        im_resize = cv2.resize(im, (256, 256))
                    cv2.imwrite(os.path.join(subdir_out, img), im_resize)
        else:
            imgs = [name for name in os.listdir(cat_dir)]
            if not os.path.isdir(cat_out):
                os.makedirs(cat_out)
            for img in imgs:
                im = cv2.imread(os.path.join(cat_dir, img))
                if im is None:
                    im_resize = np.ones((256, 256, 3), dtype=np.uint8) * 255
                else:
                    im_resize = cv2.resize(im, (256, 256))
                cv2.imwrite(os.path.join(cat_out, img), im_resize)

