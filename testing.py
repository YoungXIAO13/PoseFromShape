import argparse
import numpy as np
import os, sys
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support

sys.path.append('./auxiliary/')
from auxiliary.model import PoseEstimator, BaselineEstimator
from auxiliary.dataset import Pix3D, Pascal3D, Linemod
from auxiliary.utils import load_checkpoint
from evaluation import test_category

# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)

# model hyper-parameters
parser.add_argument('--model', type=str, default=None, help='reload model path')
parser.add_argument('--img_feature_dim', type=int, default=1024, help='feature dimension for images')
parser.add_argument('--shape_feature_dim', type=int, default=256, help='feature dimension for shapes')
parser.add_argument('--bin_size', type=int, default=15, help='bin size for the euler angle classification')

# dataset settings
parser.add_argument('--dataset', type=str, default=None,
                    choices=['ObjectNet3D', 'Pascal3D', 'ShapeNetCore'], help='dataset')
parser.add_argument('--shape_dir', type=str, default='Renders_semi_sphere',
                    choices=['Renders_semi_sphere', 'pointcloud'], help='subdirectory conatining the shape')
parser.add_argument('--shape', type=str, default='MultiView',
                    choices=['MultiView', 'PointCloud'], help='shape representation')
parser.add_argument('--view_num', type=int, default=12, help='number of render images used in each sample')
parser.add_argument('--tour', type=int, default=2, help='elevation tour for randomized references')
parser.add_argument('--random_model', action='store_true', help='whether use random model in testing')

parser.add_argument('--output_dir', type=str, default=None, help='where to save the testig results')

opt = parser.parse_args()
print(opt)
# ========================================================== #


# ================CREATE NETWORK============================ #
azi_classes, ele_classes, inp_classes = int(360 / opt.bin_size), int(180 / opt.bin_size), int(360 / opt.bin_size)

if opt.shape is None:
    model = BaselineEstimator(img_feature_dim=opt.img_feature_dim,
                              azi_classes=azi_classes, ele_classes=ele_classes, inp_classes=inp_classes)
else:
    model = PoseEstimator(shape=opt.shape, shape_feature_dim=opt.shape_feature_dim, img_feature_dim=opt.img_feature_dim,
                          azi_classes=azi_classes, ele_classes=ele_classes, inp_classes=inp_classes, view_num=opt.view_num)

model.cuda()
if not os.path.isfile(opt.model):
    raise ValueError('Non existing file: {0}'.format(opt.model))
else:
    load_checkpoint(model, opt.model)
# ========================================================== #


# =============DEFINE stuff for logs ======================= #
# write basic information into the log file
predictions_path = os.getcwd() if opt.output_dir is None else opt.output_dir
if not os.path.isdir(predictions_path):
    os.mkdir(predictions_path)
logname = os.path.join(predictions_path, 'testing_log.txt')
f = open(logname, mode='w')
f.write('\n')
f.close()
# ========================================================== #


# ================TESTING LOOP============================== #
Accs = {}
Meds = {}

root_dir = os.path.join('data', opt.dataset)
annotation_file = '{}.txt'.format(opt.dataset)
if opt.dataset == 'Pascal3D':
    test_cats = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa',
                 'train', 'tvmonitor']
    for cat in tqdm(test_cats):
        dataset_test = Pascal3D(root_dir=root_dir, annotation_file=annotation_file,
                                cat_choice=[cat], train=False, random=False, shape=opt.shape, shape_dir=opt.shape_dir,
                                view_num=opt.view_num, tour=opt.tour, random_model=opt.random_model)
        Accs[cat], Meds[cat] = test_category(opt.shape, opt.batch_size, opt.dataset, dataset_test, model, opt.bin_size,
                                             cat, predictions_path, logname)


elif opt.dataset == 'ObjectNet3D':
    test_cats = ['bed', 'bookshelf', 'calculator', 'cellphone', 'computer', 'door', 'filing_cabinet', 'guitar', 'iron',
                 'knife', 'microwave', 'pen', 'pot', 'rifle', 'shoe', 'slipper', 'stove', 'toilet', 'tub', 'wheelchair']
    for cat in tqdm(test_cats):
        dataset_test = Pascal3D(root_dir=root_dir, annotation_file=annotation_file, shape_dir=opt.shape_dir,
                                cat_choice=[cat], train=False, random=False, keypoint=True, shape=opt.shape,
                                view_num=opt.view_num, tour=opt.tour, random_model=opt.random_model)
        Accs[cat], Meds[cat] = test_category(opt.shape, opt.batch_size, opt.dataset, dataset_test, model, opt.bin_size,
                                             cat, predictions_path, logname)


elif opt.dataset == 'LineMod':
    test_cats = ['1', '2', '4', '5', '6', '8', '9', '10', '11', '12', '13', '14', '15']
    for cat in tqdm(test_cats):
        dataset_test = Linemod(root_dir=root_dir, annotation_file=annotation_file,
                               cat_choice=[int(cat)], shape=opt.shape,
                               view_num=opt.view_num, tour=opt.tour, shape_dir=opt.shape_dir)
        Accs[cat], Meds[cat] = test_category(opt.shape, opt.batch_size, opt.dataset, dataset_test, model, opt.bin_size,
                                             cat, predictions_path, logname)


elif opt.dataset == 'Pix3D':
    test_cats = ['bed', 'bookcase', 'chair', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']
    for cat in tqdm(test_cats):
        # use testing samples selected from the paper: https://arxiv.org/pdf/1811.07249.pdf
        if cat in ['bed', 'chair', 'desk']:
            annotation_file = '{}_annotation.txt'.format(cat)
            
        dataset_test = Pix3D(root_dir=root_dir, annotation_file=annotation_file,
                             cat_choice=[cat], shape=opt.shape, shape_dir=opt.shape_dir,
                             view_num=opt.view_num, tour=opt.tour, random_model=opt.random_model)
        Accs[cat], Meds[cat] = test_category(opt.shape, opt.batch_size, opt.dataset, dataset_test, model, opt.bin_size,
                                             cat, predictions_path, logname)

else:
    sys.exit(0)
# ========================================================== #


with open(logname, 'a') as f:
    f.write('Average for all categories  >>>>  Med_Err is %.2f, and Acc_pi/6 is %.2f \n' %
            (np.array(list(Meds.values())).mean(), np.array(list(Accs.values())).mean()))
