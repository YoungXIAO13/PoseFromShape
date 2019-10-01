import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def resize_padding(im, desired_size):
    # compute the new size
    old_size = im.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
    return new_im
    
    
im_dir = 'img'
mask_dir = 'mask'

cats = os.listdir(im_dir)
for cat in tqdm(cats):
    imgs = os.listdir(os.path.join(im_dir, cat))
    crop_dir = os.path.join('crop', cat)
    if not os.path.isdir(crop_dir):
        os.makedirs(crop_dir)
    
    for img in tqdm(imgs):
        im = Image.open(os.path.join(im_dir, cat, img)).convert('RGB')
        im_array = np.array(im)
        name = img[:4]
        mask = Image.open(os.path.join(mask_dir, cat, '{}.png'.format(name)))
        mask_array = np.array(mask)
        x, y, w, h = cv2.boundingRect(mask_array)
        new_array = im_array[y:y+h, x:x+w, :]
        new_im = Image.fromarray(new_array)
        crop_im = resize_padding(new_im, 224)
        crop_im.save(os.path.join(crop_dir, '{}.png'.format(name)))

