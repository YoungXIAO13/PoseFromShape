import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support
import pandas as pd
from PIL import Image, ImageFilter
import sys
sys.path.append('./auxiliary/')
from auxiliary.model import PoseEstimator
from auxiliary.utils import *


# ================SHOW RENDER=============================== #
import os,sys
import bpy
import math
from math import radians
from tqdm import tqdm


# create a lamp with an appropriate energy
def makeLamp(rad):
    # Create new lamp data block
    lamp_data = bpy.data.lamps.new(name="Lamp", type='POINT')
    lamp_data.energy = rad
    # modify the distance when the object is not normalized
    # lamp_data.distance = rad * 2.5

    # Create new object with our lamp data block
    lamp_object = bpy.data.objects.new(name="Lamp", object_data=lamp_data)

    # Link lamp object to the scene so it'll appear in this scene
    scene = bpy.context.scene
    scene.objects.link(lamp_object)
    return lamp_object


def parent_obj_to_camera(b_camera):
    # set the parenting to the origin
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty


def clean_obj_lamp_and_mesh(context):
    scene = context.scene
    objs = bpy.data.objects
    meshes = bpy.data.meshes
    for obj in objs:
        if obj.type == "MESH" or obj.type == 'LAMP':
            scene.objects.unlink(obj)
            objs.remove(obj)
    for mesh in meshes:
        meshes.remove(mesh)


def render_obj(obj, output_dir, azi, ele, rol, name='img', shape=[480, 640]):
    clean_obj_lamp_and_mesh(bpy.context)
    # Set up rendering of depth map:
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Add passes for additionally dumping albed and normals.
    bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
    bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create input render layer node
    rl = tree.nodes.new('CompositorNodeRLayers')
    map = tree.nodes.new(type="CompositorNodeMapValue")

    # Size is chosen kind of arbitrarily, try out until you're satisfied with
    # resulting depth map.
    map.offset = [-0.7]
    map.size = [0.8]
    map.use_min = True
    map.min = [0]
    map.use_max = True
    map.max = [255]
    try:
        links.new(rl.outputs['Z'], map.inputs[0])
    except KeyError:
        # some versions of blender don't like this?
        pass

    # invert = tree.nodes.new(type="CompositorNodeInvert")
    # links.new(map.outputs[0], invert.inputs[1])
    #
    # # create a file output node and set the path
    # depthFileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
    # depthFileOutput.label = 'Depth Output'
    # links.new(invert.outputs[0], depthFileOutput.inputs[0])
    #
    # scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
    # scale_normal.blend_type = 'MULTIPLY'
    # # scale_normal.use_alpha = True
    # scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    # links.new(rl.outputs['Normal'], scale_normal.inputs[1])
    #
    # bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
    # bias_normal.blend_type = 'ADD'
    # # bias_normal.use_alpha = True
    # bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    # links.new(scale_normal.outputs[0], bias_normal.inputs[1])
    #
    # normalFileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
    # normalFileOutput.label = 'Normal Output'
    # links.new(bias_normal.outputs[0], normalFileOutput.inputs[0])

    # Setting up the environment
    scene = bpy.context.scene
    scene.render.resolution_x = shape[1]
    scene.render.resolution_y = shape[0]
    scene.render.resolution_percentage = 100
    scene.render.alpha_mode = 'TRANSPARENT'

    # Camera setting
    cam = scene.objects['Camera']
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty

    # Light setting
    lamp_object = makeLamp(5)
    lamp_add = makeLamp(1)

    # Output setting
    fp = output_dir
    scene.render.image_settings.file_format = 'PNG'
    # outputs = {
    #     'depth': depthFileOutput,
    #     'normal': normalFileOutput
    # }
    # for output_node in [depthFileOutput, normalFileOutput]:
    #     output_node.base_path = ''

    # import object
    bpy.ops.import_scene.obj(filepath=obj)
    # change axis for sheep, horse obtained from Free3D
    #bpy.ops.import_scene.obj(filepath=obj, axis_forward='Y', axis_up='Z')

    # normalize it and set the center
    for object in bpy.context.scene.objects:
        if object.name in ['Camera', 'Lamp'] or object.type == 'EMPTY':
            continue
        bpy.context.scene.objects.active = object
        max_dim = max(object.dimensions)
        object.dimensions = object.dimensions / max_dim if max_dim != 0 else object.dimensions

    # Separate viewpoints on the surface of a semi-sphere of radian r
    r = 3
    scene.render.filepath = os.path.join(fp, name + 'rendering_%03d_%03d_%03d' % (int(azi), int(ele), int(rol)))
    #for k, v in outputs.items():
    #    v.file_slots[0].path = os.path.join(fn, 'rendering_%03d_%03d_%s' % (int(azi), int(ele), k))


    # transform Euler angles from degrees into radians
    #azi = radians((azi - 90) % 360) # if lucy, rotate 90 degrees
    azi = radians(azi)
    ele = radians(ele)
    rol = radians(rol)

    # if bunny or dragon, change the sign of loc_x and loc_y
    loc_y = r * math.cos(ele) * math.cos(azi)
    loc_x = r * math.cos(ele) * math.sin(azi)
    loc_z = r * math.sin(ele)
    cam.location = (loc_x, loc_y, loc_z + 0.5)
    lamp_object.location = (loc_x, loc_y, 10)
    lamp_add.location = (loc_x, loc_y, -10)

    # Change the in-plane rotation
    cam_ob = bpy.context.scene.camera
    bpy.context.scene.objects.active = cam_ob  # select the camera object
    distance = np.sqrt(loc_x ** 2 + loc_y ** 2 + loc_z ** 2)
    bpy.ops.transform.rotate(value=rol, axis=(loc_x / distance, loc_y / distance, loc_z / distance),
                             constraint_axis=(False, False, False), constraint_orientation='GLOBAL', mirror=False,
                             proportional='DISABLED', proportional_edit_falloff='SMOOTH',
                             proportional_size=1)

    bpy.ops.render.render(write_still=True)
# ========================================================== #


# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()

parser.add_argument('--shape', type=str, default=None, help='shape representation')
parser.add_argument('--model', type=str, default=None,  help='optional reload model path')
parser.add_argument('--channels', type=int, default=3, help='input channels for the ResNet')
parser.add_argument('--separate_branch', action='store_true', help='use separated branch for classification and regression network')
parser.add_argument('--num_render', type=int, default=12,  help='number of render images used in each sample')
parser.add_argument('--img_feature_dim', type=int, default=1024,  help='feature dimension for textured images')
parser.add_argument('--shape_feature_dim', type=int, default=256,  help='feature dimension for non-textured images')
parser.add_argument('--azi_classes', type=int, default=24,  help='number of class for azimuth')
parser.add_argument('--ele_classes', type=int, default=12,  help='number of class for elevation')
parser.add_argument('--inp_classes', type=int, default=24,  help='number of class for inplane_rotation')
parser.add_argument('--features', type=int, default=64, help='number of inplanes for the ResNet')
parser.add_argument('--input_dim', type=int, default=224, help='input image dimension')

parser.add_argument('--image_path', type=str, default=None, help='real images path')
parser.add_argument('--render_path', type=str, default=None, help='render images path')
parser.add_argument('--obj_path', type=str, default=None, help='obj path')

opt = parser.parse_args()
print(opt)
# ========================================================== #


# ================CREATE NETWORK============================ #
model = PoseEstimator(shape=opt.shape, shape_feature_dim=opt.shape_feature_dim, img_feature_dim=opt.img_feature_dim,
                      azi_classes=opt.azi_classes, ele_classes=opt.ele_classes, inp_classes=opt.inp_classes,
                      render_number=opt.num_render, separate_branch=opt.separate_branch, channels=opt.channels)
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

# define data preprocessing for render images
render_transform = transforms.ToTensor()

# load render images in a Tensor of size K*C*H*W
render_names = [name for name in os.listdir(opt.render_path)]
render_names.sort()
step = int(36 / (opt.num_render / 2))
render_names = render_names[0:72:step]
for i in range(0, len(render_names)):
    render = Image.open(os.path.join(opt.render_path, render_names[i])).convert('RGB')
    render = render_transform(render)
    if i == 0:
        renders = render.unsqueeze(0)
    else:
        renders = torch.cat((renders, render.unsqueeze(0)), 0)

K, C, H, W = renders.size()
renders = renders.view(1, K, C, H, W)
# ========================================================== #


# ================TESTING LOOP============================== #
#csv_file = os.path.join(opt.image_path, 'results.txt')
#if not os.path.exists(csv_file):
#    with open(csv_file, 'a') as f:
#        f.write('image_name' + ',' + 'proba_azi' + ',' + 'proba_ele' + ',' + 'proba_inp' + ',' + 'proba' + ',' + 'entropy' + ',' + 'azimuth' + ',' + 'elevation' + ',' + 'inplane_rotation' + '\n')

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
        im = resize_padding(im, 224)
        im = data_validating(im)

        # resize it as a batch
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

        # compute the entropy of orientation classification
        prob_azi = torch.nn.functional.softmax(out_azi.view(-1))
        entr_azi = -torch.sum(prob_azi * prob_azi.log()).item()
        prob_azi = prob_azi[pred_azi.long()].item()

        prob_ele = torch.nn.functional.softmax(out_ele.view(-1))
        entr_ele = -torch.sum(prob_ele * prob_ele.log()).item()
        prob_ele = prob_ele[pred_ele.long()].item()

        prob_rol = torch.nn.functional.softmax(out_rol.view(-1))
        entr_rol = -torch.sum(prob_rol * prob_rol.log()).item()
        prob_rol = prob_rol[pred_rol.long()].item()

        prob = ((prob_azi + prob_ele + prob_rol) / 3)
        entropy = ((entr_azi + entr_ele + entr_rol) / 3)

        out_reg = out[3].sigmoid().squeeze()

        azi = ((pred_azi.float() + out_reg[0]) * (360. / opt.azi_classes)).item()
        ele = (((pred_ele.float() + out_reg[1]) * (360. / opt.azi_classes)) - 90).item()
        rol = (((pred_rol.float() + out_reg[2]) * (360. / opt.azi_classes)) - 180).item()

        # render the object under predicted pose
        output_path = opt.image_path
        # redirect output to log file
        logfile = 'render_random.log'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

        img_name = img_name.split(".")[0]
        #with open(csv_file, 'a') as f:
        #    f.write(img_name + ',' + str(prob_azi) + ',' + str(prob_ele) + ',' + str(prob_rol) + ',' + str(prob) + ',' + str(entropy) + ',' + str(int(azi)) + ',' + str(int(ele)) + ',' + str(int(rol)) + '\n')

        render_obj(opt.obj_path, output_path, azi, ele, rol, img_name)

        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)
# ========================================================== #
