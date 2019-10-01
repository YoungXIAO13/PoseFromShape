import torch.utils.data as data
import os
from os.path import join, basename
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFilter
import pandas as pd
import cv2
import math
import pymesh


# Lighting noise transform
class TransLightning(object):
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


# ImageNet statistics
imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([[-0.5675, 0.7192, 0.4009],
                                [-0.5808, -0.0045, -0.8140],
                                [-0.5836, -0.6948, 0.4203],
                                ])
    }


# Define normalization and random disturb for input image
disturb = TransLightning(0.1, imagenet_pca['eigval'], imagenet_pca['eigvec'])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# Crop the image using random bounding box with IoU >= 0.7 compared with the ground truth
def random_crop(im, x, y, w, h):
    left = max(0, x + int(np.random.uniform(-0.1, 0.1) * w))
    upper = max(0, y + int(np.random.uniform(-0.1, 0.1) * h))
    right = min(im.size[0], x + int(np.random.uniform(0.9, 1.1) * w))
    lower = min(im.size[1], y + int(np.random.uniform(0.9, 1.1) * h))
    im_crop = im.crop((left, upper, right, lower))
    return im_crop


def resize_pad(im, dim):
    w, h = im.size
    im = transforms.functional.resize(im, int(dim * min(w, h) / max(w, h)))
    left = int(np.ceil((dim - im.size[0]) / 2))
    right = int(np.floor((dim - im.size[0]) / 2))
    top = int(np.ceil((dim - im.size[1]) / 2))
    bottom = int(np.floor((dim - im.size[1]) / 2))
    im = transforms.functional.pad(im, (left, top, right, bottom))
    return im


def resize_padding(im, desired_size):
    # compute the new size
    old_size = im.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_im


def read_multiviwes(render_transform, render_example_path, view_num, tour, mutation):
    """
    Read multi view rendered images from the target path
    :param render_transform: image processing applied to the rendered image
    :param render_example_path: folder containing the rendered images for training example
    :param view_num: number of rendered images used as 3D shape representation
    :param tour: number of elevations of the rendered images
    :param mutation: randomization with respect to the canonical view in term of azimuth
    :return: shape tensor of dimension (view_num, C, H, W)
    """
    render_names = [name for name in os.listdir(render_example_path)]
    render_names.sort()
    step = int(72 / (view_num / tour))
    renders_low = np.linspace(0, 71, 72, dtype='int')
    renders_mid = renders_low + 72
    renders_up = renders_mid + 72
    if tour == 1:
        render_ids = np.concatenate((renders_mid[mutation:], renders_mid[:mutation]))[::step]
    elif tour == 2:
        render_ids = np.concatenate((np.concatenate((renders_low[mutation:], renders_low[:mutation]))[::step],
                                     np.concatenate((renders_mid[mutation:], renders_mid[:mutation]))[::step]))
    else:
        render_ids = np.concatenate((np.concatenate((renders_low[mutation:], renders_low[:mutation]))[::step],
                                     np.concatenate((renders_mid[mutation:], renders_mid[:mutation]))[::step],
                                     np.concatenate((renders_up[mutation:], renders_up[:mutation]))[::step]))

    # load multi views and concatenate them into a tensor
    renders = []
    for i in range(0, len(render_ids)):
        render = Image.open(os.path.join(render_example_path, render_names[render_ids[i]]))
        render = render.convert('RGB')
        render = render_transform(render)
        renders.append(render.unsqueeze(0))

    return torch.cat(renders, 0)


def read_pointcloud(model_path, point_num, rotation=0):
    """
    Read point cloud from the target path
    :param model_path: file path for the point cloud
    :param point_num: input point number of the point cloud
    :param rotation: randomization with respect to the canonical view in term of azimuth
    :return: shape tensor
    """
    # read in original point cloud
    point_cloud_raw = pymesh.load_mesh(model_path).vertices

    # randomly select a fix number of points on the surface
    point_subset = np.random.choice(point_cloud_raw.shape[0], point_num, replace=False)
    point_cloud = point_cloud_raw[point_subset]

    # apply the random rotation on the point cloud
    if rotation != 0:
        alpha = math.radians(rotation)
        rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha), 0.],
                               [np.sin(alpha), np.cos(alpha), 0.],
                               [0., 0., 1.]])
        point_cloud = np.matmul(point_cloud, rot_matrix.transpose())

    point_cloud = torch.from_numpy(point_cloud.transpose()).float()

    # normalize the point cloud into [0, 1]
    point_cloud = point_cloud - torch.min(point_cloud)
    point_cloud = point_cloud / torch.max(point_cloud)

    return point_cloud


# ================================================= #
# Datasets used for training
# ================================================= #
class Pascal3D(data.Dataset):
    def __init__(self,
                 root_dir, annotation_file, input_dim=224, shape='MultiView', shape_dir='Renders_semi_sphere',
                 random=False, novel=True, keypoint=False, train=True, cat_choice=None, random_model=False,
                 view_num=12, tour=2, random_range=0, point_num=2500):

        self.root_dir = root_dir
        self.input_dim = input_dim
        self.shape = shape
        self.shape_dir = shape_dir
        self.tour = tour
        self.view_num = view_num
        self.point_num = point_num
        self.train = train
        self.random = random
        self.random_range = random_range
        self.random_model = random_model
        self.bad_cats = ['ashtray', 'basket', 'bottle', 'bucket', 'can', 'cap', 'cup', 'fire_extinguisher', 'fish_tank',
                         'flashlight', 'helmet', 'jar', 'paintbrush', 'pen', 'pencil', 'plate', 'pot', 'road_pole',
                         'screwdriver', 'toothbrush', 'trash_bin', 'trophy']

        # load the data frame for annotations
        frame = pd.read_csv(os.path.join(root_dir, annotation_file))
        frame = frame[frame.elevation != 90]
        frame = frame[frame.difficult == 0]
        if annotation_file == 'ObjectNet3D.txt':
            if keypoint:
                frame = frame[frame.has_keypoints == 1]
                frame = frame[frame.truncated == 0]
                frame = frame[frame.occluded == 0]
            frame.azimuth = (360. + frame.azimuth) % 360
        if train:
            frame = frame[frame.set == 'train']
        else:
            frame = frame[frame.set == 'val']
            frame = frame[frame.truncated == 0]
            frame = frame[frame.occluded == 0]

        # choose cats for Object3D
        if cat_choice is not None:
            if train:
                frame = frame[~frame.cat.isin(cat_choice)] if novel else frame
            else:
                frame = frame[frame.cat.isin(cat_choice)]
        self.annotation_frame = frame

        # define data augmentation and preprocessing for RGB images in training
        self.im_augmentation = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.ToTensor(), normalize, disturb])

        # define data preprocessing for RGB images in validation
        self.im_transform = transforms.Compose([transforms.ToTensor(), normalize])

        # define data preprocessing for rendered multi view images
        self.render_path = 'crop'
        self.render_transform = transforms.ToTensor()
        if input_dim != 224:
            self.render_transform = transforms.Compose([transforms.Resize(input_dim), transforms.ToTensor()])

    def __len__(self):
        return len(self.annotation_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotation_frame.iloc[idx]['im_path'])
        cat = self.annotation_frame.iloc[idx]['cat']
        cad_index = self.annotation_frame.iloc[idx]['cad_index']

        # select a random shape from the same category in testing
        if self.random_model:
            df_cat = self.annotation_frame[self.annotation_frame.cat == cat]
            df_cat = df_cat[df_cat.cad_index != cad_index]
            random_idx = np.random.randint(len(df_cat))
            cad_index = self.annotation_frame.iloc[random_idx]['cad_index']

        left = self.annotation_frame.iloc[idx]['left']
        upper = self.annotation_frame.iloc[idx]['upper']
        right = self.annotation_frame.iloc[idx]['right']
        lower = self.annotation_frame.iloc[idx]['lower']

        # use continue viewpoint annotation
        label = self.annotation_frame.iloc[idx, 9:12].values

        # load real images in a Tensor of size C*H*W
        im = Image.open(img_name).convert('RGB')

        if self.train:
            # Gaussian blur
            if min(right - left, lower - upper) > 224 and np.random.random() < 0.3:
                im = im.filter(ImageFilter.GaussianBlur(3))

            # crop the original image with 2D bounding box jittering
            im = random_crop(im, left, upper, right - left, lower - upper)

            # Horizontal flip
            if np.random.random() > 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                label[0] = 360 - label[0]
                label[2] = -label[2]

            # Random rotation
            if np.random.random() > 0.5:
                r = max(-60, min(60, np.random.randn() * 30))
                im = im.rotate(r)
                label[2] = label[2] + r
                label[2] += 360 if label[2] < -180 else (-360 if label[2] > 180 else 0)

            # pad it to the desired size
            im = resize_pad(im, self.input_dim)
            im = self.im_augmentation(im)
        else:
            im = im.crop((left, upper, right, lower))
            im = resize_pad(im, self.input_dim)
            im = self.im_transform(im)

        label[0] = (360. - label[0]) % 360.
        label[1] = label[1] + 90.
        label[2] = (label[2] + 180.) % 360.
        label = label.astype('int')

        if self.shape is None:
            label = torch.from_numpy(label).long()
            return im, label

        # randomize the canonical frame in azimuth
        # range_0: [-45, 45]; range_1: [-90, 90]; range_2: [-180, 180]
        if self.random and cat not in self.bad_cats:
            mutation = np.random.randint(-8, 9) % 72 if self.random_range == 0 else \
                (np.random.randint(-17, 18) % 72 if self.random_range == 1 else np.random.randint(0, 72))
            label[0] = (label[0] - mutation * 5) % 360
        else:
            mutation = 0

        if self.shape == 'MultiView':
            # load render images in a Tensor of size K*C*H*W
            render_example_path = os.path.join(self.root_dir, self.shape_dir, cat, '%02d' % cad_index, self.render_path)
            renders = read_multiviwes(self.render_transform, render_example_path, self.view_num, self.tour, mutation)

            label = torch.from_numpy(label).long()
            return im, renders, label

        if self.shape == 'PointCloud':
            example_path = os.path.join(self.root_dir, self.shape_dir, cat, '%02d' % cad_index, 'compressed.ply')
            point_cloud = read_pointcloud(example_path, self.point_num, mutation)

            return im, point_cloud, label


class ShapeNet(data.Dataset):
    def __init__(self, root_dir, annotation_file, bg_dir, bg_list='SUN_database.txt',
                 input_dim=224, model_number=200, novel=False,
                 shape='MultiView', shape_dir='Renders_semi_sphere',
                 view_num=12, tour=2, random_range=0, point_num=2500,
                 cat_choice=None, train=True, random=False):
        self.root_dir = root_dir
        self.input_dim = input_dim
        self.bg_dir = bg_dir
        self.bg_list = pd.read_csv(os.path.join(self.bg_dir, bg_list))
        self.shape = shape
        self.shape_dir = shape_dir
        self.view_num = view_num
        self.point_num = point_num
        self.tour = tour
        self.random_range = random_range
        self.train = train
        self.random = random

        # load the appropriate data frame for annotations
        frame = pd.read_csv(os.path.join(root_dir, annotation_file))
        if cat_choice is not None:
            if train:
                frame = frame[~frame.cat_id.isin(cat_choice)] if novel else frame
            else:
                frame = frame[frame.cat_id.isin(cat_choice)]
        cats = np.unique(frame.cat_id)
        for i in range(0, len(cats)):
            frame_cat = frame[frame.cat_id == cats[i]]
            examples = list(np.unique(frame_cat.example_id))
            if len(examples) > model_number:
                examples = examples[:model_number]
            if i == 0:
                new_frame = frame_cat[frame_cat.example_id.isin(examples)]
            else:
                new_frame = pd.concat([new_frame, frame_cat[frame_cat.example_id.isin(examples)]])
        self.annotation_frame = new_frame

        # define data augmentation and preprocessing for RGB images
        im_augmentation = transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                              transforms.RandomCrop(224)])
        im_transform = transforms.CenterCrop(224)
        if input_dim != 224:
            im_augmentation = transforms.Compose([im_augmentation, transforms.Resize(input_dim)])
            im_transform = transforms.Compose([im_transform, transforms.Resize(input_dim)])
        self.im_augmentation = transforms.Compose([im_augmentation, transforms.ToTensor(), normalize, disturb])
        self.im_transform = transforms.Compose([im_transform, transforms.ToTensor(), normalize])

        # define data preprocessing for rendered images
        self.render_path = 'crop'
        self.render_transform = transforms.ToTensor()
        if input_dim != 224:
            self.render_transform = transforms.Compose([transforms.Resize(input_dim), transforms.ToTensor()])

    def __len__(self):
        return len(self.annotation_frame)

    def __getitem__(self, idx):
        cat_id = self.annotation_frame.iloc[idx]['cat_id']
        example_id = self.annotation_frame.iloc[idx]['example_id']
        label = self.annotation_frame.iloc[idx, 3:].values
        label = np.append(label, [0])
        label = label.astype('int')

        # load render images
        im_render = Image.open(os.path.join(self.root_dir, self.annotation_frame.iloc[idx]['image_path']))

        # random rotation
        r = max(-45, min(45, np.random.randn() * 15))
        im_render = im_render.rotate(r)
        label[-1] = label[-1] + r

        # load background images and composite it with render images
        bg = cv2.imread(os.path.join(self.bg_dir, self.bg_list.iloc[np.random.randint(len(self.bg_list)), 1])).copy()
        if bg is None or bg.shape[0:2] != im_render.size:
            bg = np.ones((im_render.size[1], im_render.size[0], 3), dtype=np.uint8) * 255
        im_composite = bg[:, :, ::-1]
        im_composite = Image.fromarray(im_composite)
        im_composite.paste(im_render, (0, 0), im_render)

        if self.train:
            # Gaussian blur
            if np.random.random() < 0.3:
                im_composite = im_composite.filter(ImageFilter.GaussianBlur(3))

            # Horizontal flip
            if np.random.random() > 0.5:
                im_composite = im_composite.transpose(Image.FLIP_LEFT_RIGHT)
                label[0] = (360 - label[0]) % 360
                label[-1] = -label[-1]

            im = self.im_augmentation(im_composite)
        else:
            im = self.im_transform(im_composite)

        # change the original label to classification label
        label[1] = label[1] + 90.
        label[-1] = label[-1] + 180

        # load the correct data according to selected shape representation
        if self.shape is None:
            label = torch.from_numpy(label).long()
            return im, label

        # randomize the canonical frame in azimuth
        # range_0: [-45, 45]; range_1: [-90, 90]; range_2: [-180, 180]
        if self.random:
            mutation = np.random.randint(-8, 9) % 72 if self.random_range == 0 else \
                (np.random.randint(-17, 18) % 72 if self.random_range == 1 else np.random.randint(0, 72))
            label[0] = (label[0] - mutation * 5) % 360
        else:
            mutation = 0

        if self.shape == 'MultiView':
            # load render images in a Tensor of size K*C*H*W
            render_example_path = os.path.join(self.root_dir, self.shape_dir, str('%08d' % cat_id), example_id,
                                               self.render_path)
            renders = read_multiviwes(self.render_transform, render_example_path, self.view_num, self.tour,
                                      mutation)

            label = torch.from_numpy(label).long()
            return im, renders, label

        if self.shape == 'PointCloud':
            example_path = os.path.join(self.root_dir, self.shape_dir, cat, str('%08d' % cat_id), example_id, 'compressed.ply')
            point_cloud = read_pointcloud(example_path, self.point_num, mutation)

            return im, point_cloud, label


# ================================================= #
# Datasets only used for evaluation
# ================================================= #
class Pix3D(data.Dataset):
    def __init__(self,
                 root_dir, annotation_file, input_dim=224, shape='MultiView',
                 cat_choice=None, random_model=False,
                 shape_dir='Renders_semi_sphere', view_num=12, tour=2):
        self.root_dir = root_dir
        self.shape = shape
        self.shape_dir = shape_dir
        self.view_num = view_num
        self.tour = tour
        self.random_model = random_model
        self.render_path = 'crop'
        self.render_transform = transforms.ToTensor()
        if input_dim != 224:
            self.render_transform = transforms.Compose([transforms.Resize(input_dim), transforms.ToTensor()])

        # load the appropriate data frame for annotations
        frame = pd.read_csv(os.path.join(root_dir, annotation_file))
        frame = frame[frame.truncated == False]
        frame = frame[frame.occluded == False]
        frame = frame[frame.slightly_occluded == False]
        frame.elevation = frame.elevation + 90.
        frame.inplane_rotation = (frame.inplane_rotation * 180. / np.pi) + 180.
        if cat_choice is not None:
            frame = frame[frame.cat_id.isin(cat_choice)]
        self.annotation_frame = frame

        # define data preprocessing for query images
        self.im_transform = transforms.Compose([transforms.ToTensor(), normalize])
        if input_dim != 224:
            self.im_transform = transforms.Compose([transforms.Resize(input_dim), self.im_transform])

    def __len__(self):
        return len(self.annotation_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotation_frame.iloc[idx]['image_path'])
        cat_id = self.annotation_frame.iloc[idx]['cat_id']
        example_id = self.annotation_frame.iloc[idx]['example_id']
        model_name = self.annotation_frame.iloc[idx]['model_name']
        if self.random_model:
            df_cat = self.annotation_frame[self.annotation_frame.cat_id == cat_id]
            df_cat = df_cat[df_cat.example_id != example_id]
            random_idx = np.random.randint(len(df_cat))
            example_id = self.annotation_frame.iloc[random_idx]['example_id']
            model_name = self.annotation_frame.iloc[random_idx]['model_name']
        label = self.annotation_frame.iloc[idx, 9:].values
        label = label.astype('int')
        label = torch.from_numpy(label).long()

        # load real images in a Tensor of size C*H*W
        im = Image.open(img_name).convert('RGB')
        im = self.im_transform(im)
        
        if self.shape is None:
            return im, label

        elif self.shape == 'MultiView':
            # load render images in a Tensor of size K*C*H*W
            if model_name == 'model':
                render_example_path = os.path.join(self.root_dir, self.shape_dir, cat_id, example_id, self.render_path)
            else:
                render_example_path = os.path.join(self.root_dir, self.shape_dir, cat_id, example_id, model_name, self.render_path)

            # read multiview rendered images
            renders = read_multiviwes(self.render_transform, render_example_path, self.view_num, self.tour, 0)

            return im, renders, label


class Linemod(data.Dataset):
    def __init__(self,
                 root_dir, annotation_file, input_dim=224, shape='MultiView', cat_choice=None,
                 shape_dir='Renders_semi_sphere', view_num=12, tour=2):
        self.root_dir = root_dir
        self.shape = shape
        self.shape_dir = shape_dir
        self.view_num = view_num
        self.tour = tour
        self.render_path = 'crop'
        self.render_transform = transforms.ToTensor()
        if input_dim != 224:
            self.render_transform = transforms.Compose([transforms.Resize(input_dim), transforms.ToTensor()])

        # load the appropriate data frame for annotations
        frame = pd.read_csv(os.path.join(root_dir, annotation_file))

        # choose cats
        if cat_choice is not None:
            frame = frame[frame.obj_id.isin(cat_choice)]
        self.annotation_frame = frame

        self.im_transform = transforms.Compose([transforms.ToTensor(), normalize])
        if input_dim != 224:
            self.im_transform = transforms.Compose([transforms.Resize(input_dim), self.im_transform])

    def __len__(self):
        return len(self.annotation_frame)

    def __getitem__(self, idx):
        obj_id = self.annotation_frame.iloc[idx]['obj_id']
        img_name = os.path.join(self.root_dir, self.annotation_frame.iloc[idx]['image_path'])
        x = self.annotation_frame.iloc[idx]['x']
        y = self.annotation_frame.iloc[idx]['y']
        w = self.annotation_frame.iloc[idx]['w']
        h = self.annotation_frame.iloc[idx]['h']

        # use continue viewpoint annotation
        label = self.annotation_frame.iloc[idx, 6:].values

        # load real images in a Tensor of size C*H*W
        im = Image.open(img_name)
        im = im.crop((x, y, x + w, y + h))
        im = resize_pad(im, 224)
        im = self.im_transform(im)

        label[1] = label[1] + 90.
        label[2] = (-label[2] + 180.) % 360.
        label = label.astype('int')
        label = torch.from_numpy(label).long()

        if self.shape is None:
            return im, label

        elif self.shape == 'MultiView':
            # load render images in a Tensor of size K*C*H*W
            render_example_path = os.path.join(self.root_dir, self.shape_dir, '%02d' % obj_id, self.render_path)

            # read multiview rendered images
            renders = read_multiviwes(self.render_transform, render_example_path, self.view_num, self.tour, 0)

            return im, renders, label


if __name__ == "__main__":
    #d = Pix3d(shape='MultiView', cat_choice=None, view_num=12, tour=2)

    #d = ShapeNet(annotation_file='annotation_all_texture.txt', shape='MultiView', cat_choice=test_cats, novel=True)

    #d = Linemod(cat_choice=[1], shape='MultiView')

    root_dir = '/home/xiao/Datasets/Pascal3D'
    annotation_file = 'Pascal3D.txt'
    d = Pascal3D(root_dir, annotation_file, shape='PointCloud', shape_dir='pointcloud')

    print('length is %d' % len(d))

    from torch.utils.data import DataLoader
    import sys
    import time
    test_loader = DataLoader(d, batch_size=4, shuffle=False)
    begin = time.time()
    for i, data in enumerate(test_loader):
        im, shape, label = data
        print(time.time() - begin)
        if i == 0:
            print(im.size(), shape.size(), label.size())
            print(label)
            sys.exit()
