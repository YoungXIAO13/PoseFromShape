import argparse
import os, sys
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support

sys.path.append('./auxiliary/')
from auxiliary.model import PoseEstimator
from auxiliary.dataset import read_multiviwes, resize_padding
from data.render_utils import render_obj

# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None,  help='optional reload model path')
parser.add_argument('--shape', type=str, default='MultiView', help='shape representation')

# model hyper-parameters
parser.add_argument('--view_num', type=int, default=12,  help='number of render images used in each sample')
parser.add_argument('--tour', type=int, default=2, help='elevation tour for randomized references')
parser.add_argument('--img_feature_dim', type=int, default=1024,  help='feature dimension for textured images')
parser.add_argument('--shape_feature_dim', type=int, default=256,  help='feature dimension for non-textured images')
parser.add_argument('--bin_size', type=int, default=15, help='bin size for the euler angle classification')
parser.add_argument('--input_dim', type=int, default=224, help='input image dimension')

# path to load the necessary input for network inference
parser.add_argument('--image_path', type=str, default=None, help='real images path')
parser.add_argument('--render_path', type=str, default=None, help='render images path')
parser.add_argument('--obj_path', type=str, default=None, help='obj path')

opt = parser.parse_args()
print(opt)
# ========================================================== #


# ================CREATE NETWORK============================ #
azi_classes, ele_classes, inp_classes = int(360 / opt.bin_size), int(180 / opt.bin_size), int(360 / opt.bin_size)

model = PoseEstimator(shape=opt.shape, shape_feature_dim=opt.shape_feature_dim, img_feature_dim=opt.img_feature_dim,
                      azi_classes=azi_classes, ele_classes=ele_classes, inp_classes=inp_classes, view_num=opt.view_num)

model.cuda()
if opt.model is not None:
    checkpoint = torch.load(opt.model, map_location=lambda storage, loc: storage.cuda())
    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Previous weight loaded')
model.eval()
# ========================================================== #


# ==================INPUT IMAGE AND RENDERS================= #
# define data preprocessing for real images in validation
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data_validating = transforms.Compose([transforms.ToTensor(), normalize])

# load render images in a Tensor of size K*C*H*W
render_transform = transforms.ToTensor()
renders = read_multiviwes(render_transform, opt.render_path, opt.view_num, opt.tour, False)

K, C, H, W = renders.size()
renders = renders.view(1, K, C, H, W)
# ========================================================== #


# ================TESTING LOOP============================== #
imgs = os.listdir(opt.image_path)
imgs.sort()
with torch.no_grad():
    for img_name in tqdm(imgs):
        im = Image.open(os.path.join(opt.image_path, img_name))
        if im.mode == 'RGBA':
            # load background images and composite it with render images
            im_composite = np.ones((im.size[1], im.size[0], 3), dtype=np.uint8) * 255
            im_composite = Image.fromarray(im_composite)
            im_composite.paste(im, (0, 0), im)

            bbox = im.getbbox()
            im = im_composite.crop(bbox)
            im = im.convert('RGB')
        im = resize_padding(im, 224).convert('RGB')
        im = data_validating(im)
        im = im.view(1, C, H, W)

        im = im.cuda()
        renders = renders.cuda()

        out = model.forward(im, renders)
        out_azi = out[0]
        out_ele = out[1]
        out_rol = out[2]

        _, pred_azi = out_azi.topk(1, 1, True, True)
        _, pred_ele = out_ele.topk(1, 1, True, True)
        _, pred_rol = out_rol.topk(1, 1, True, True)

        out_reg = out[3].sigmoid().squeeze()

        azi = ((pred_azi.float() + out_reg[0]) * opt.bin_size).item()
        ele = (((pred_ele.float() + out_reg[1]) * opt.bin_size) - 90).item()
        rol = (((pred_rol.float() + out_reg[2]) * opt.bin_size) - 180).item()

        # render the object under predicted pose
        output_path = opt.image_path
        img_name = img_name.split(".")[0]
        render_obj(opt.obj_path, output_path, azi, ele, rol, img_name)

# ========================================================== #
