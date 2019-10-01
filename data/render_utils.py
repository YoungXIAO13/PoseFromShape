"""
A simple script that uses bpy to render views of a single object by
move the camera around it.

Original source:
https://github.com/panmari/stanford-shapenet-renderer
"""

import os
import bpy
import math
from math import radians
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2

def resize_padding(im, desired_size):
    # compute the new size
    old_size = im.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_im = Image.new("RGBA", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_im
    
    
def resize_padding_v2(im, desired_size_in, desired_size_out):
    # compute the new size
    old_size = im.size
    ratio = float(desired_size_in)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_im = Image.new("RGBA", (desired_size_out, desired_size_out))
    new_im.paste(im, ((desired_size_out - new_size[0]) // 2, (desired_size_out - new_size[1]) // 2))
    return new_im
    
    
# create a lamp with an appropriate energy
def makeLamp(lamp_name, rad):
    # Create new lamp data block
    lamp_data = bpy.data.lamps.new(name=lamp_name, type='POINT')
    lamp_data.energy = rad
    # modify the distance when the object is not normalized
    # lamp_data.distance = rad * 2.5

    # Create new object with our lamp data block
    lamp_object = bpy.data.objects.new(name=lamp_name, object_data=lamp_data)

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


def render_obj_grid(obj, output_dir, shape=[256, 256], step=30, light_main=5, light_add=1, r=2, normalize=False, forward=None, up=None):
    clean_obj_lamp_and_mesh(bpy.context)
    # Set up rendering of depth map:
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create input render layer node
    rl = tree.nodes.new('CompositorNodeRLayers')
    map = tree.nodes.new(type="CompositorNodeMapValue")

    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
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

    invert = tree.nodes.new(type="CompositorNodeInvert")
    links.new(map.outputs[0], invert.inputs[1])

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
    lamp_object = makeLamp('Lamp1', light_main)
    lamp_add = makeLamp('Lamp2', light_add)

    # Output setting
    fp = os.path.join(output_dir, 'no_texture')
    crop_dir = os.path.join(output_dir, 'crop')
    if not os.path.isdir(crop_dir):os.makedirs(crop_dir)
    scene.render.image_settings.file_format = 'PNG'

    # import object
    if forward is not None and up is not None:
        bpy.ops.import_scene.obj(filepath=obj, axis_forward=forward, axis_up=up)
    else:
        bpy.ops.import_scene.obj(filepath=obj)

    # normalize the object
    if normalize:
        for object in bpy.context.scene.objects:
            if object.name in ['Camera', 'Lamp'] or object.type in ['EMPTY', 'LAMP']:
                continue
            bpy.context.scene.objects.active = object
            max_dim = max(object.dimensions)
            object.dimensions = object.dimensions / max_dim if max_dim != 0 else object.dimensions
    
    # Separate viewpoints on the surface of a semi-sphere of radian r
    n_azi = int(360 / 5)  # one render image every 5 degrees
    n_view = n_azi * int(90 / step)  # numbe of tours depending on the elevation step

    for i in range(0, n_view):
        azi = (i * 5) % 360
        ele = (i // n_azi) * step
        scene.render.filepath = os.path.join(fp, 'rendering_%03d_%03d' % (ele, azi))
    
        loc_y = r * math.cos(radians(ele)) * math.cos(radians(azi))
        loc_x = r * math.cos(radians(ele)) * math.sin(radians(azi))
        loc_z = r * math.sin(radians(ele))
        cam.location = (loc_x, loc_y, loc_z)
        lamp_object.location = (loc_x, loc_y, 10)
        lamp_add.location = (loc_x, loc_y, -10)

        # render image
        bpy.ops.render.render(write_still=True)
        
        # crop rendered images
        im_path = 'rendering_%03d_%03d.png' % (ele, azi)
        im = Image.open(os.path.join(fp, im_path)).copy()
        bbox = im.getbbox()
        im = im.crop(bbox)
        im_new = resize_padding(im, 224)
        im_new.save(os.path.join(crop_dir, im_path))
        
        
def render_obj_with_view(obj, output_dir, csv_file, texture_img, views=20, shape=[512, 512]):
    # Clean old objects
    clean_obj_lamp_and_mesh(bpy.context)
    
    # import object
    bpy.ops.import_scene.obj(filepath=obj)
        
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

    # Output setting
    fp = os.path.join(output_dir, 'renders')
    fm = os.path.join(output_dir, 'masks')
    fi = os.path.join(output_dir, 'images')
    if not os.path.isdir(fm):
        os.makedirs(fm)
    if not os.path.isdir(fi):
        os.makedirs(fi)
    scene.render.image_settings.file_format = 'PNG'

    # Light setting and Camera radian setting
    lamp_object = makeLamp('Lamp1', 3)
    lamp_add = makeLamp('Lamp2', 3 * 0.1)

    # Random textures and backgrounds generation with different viewpoints
    energies = np.random.rand(views) + .1
    r = 1.5
    azis = 360.0 * np.random.rand(views)
    azis[azis == 360.0] = 0.0  # prevent symmetry
    eles = 180.0 * np.arccos(2 * np.random.rand(views) - 1) / np.pi
    eles = np.abs(eles - 90.0)
    eles[eles == 90.0] = 89.0  # prevent absolute bird-view

    for object in bpy.context.scene.objects:
        if object.name in ['Camera', 'Lamp'] or object.type in ['EMPTY', 'LAMP']:
            continue
        mat = bpy.data.materials.new(name='Material')
        mat.diffuse_color = tuple(np.random.rand(3))
        tex = bpy.data.textures.new('UVMapping', 'IMAGE')
        tex.image = bpy.data.images.load(texture_img)
        slot = mat.texture_slots.add()
        slot.texture = tex
        if object.data.materials:
            for i in range(len(object.data.materials)):
                object.data.materials[i] = mat
        else:
            object.data.materials.append(mat)

    # Render images and crop the object and resize it to desired BBox size
    for n in range(0, views):
        loc_y = r * math.cos(radians(eles[n])) * math.cos(radians(azis[n]))
        loc_x = r * math.cos(radians(eles[n])) * math.sin(radians(azis[n]))
        loc_z = r * math.sin(radians(eles[n]))
        cam.location = (loc_x, loc_y, loc_z)
        lamp_object.location = (loc_x, loc_y, 10)
        lamp_add.location = (loc_x, loc_y, -10)

        # Modify the lightness
        for object in bpy.context.scene.objects:
            if not object.type == 'LAMP':
                continue
            if object.name == 'Lamp1':
                object.data.energy = 3 * energies[n] + 2.
            else:
                object.data.energy = 3 * energies[n] * 0.3 + .2
    
        # Render the object
        scene.render.filepath = fp + '/%03d_%03d_%03d' % (n, int(eles[n]), int(azis[n]))
        bpy.ops.render.render(write_still=True)

        # obtain the object mask of the rendered object
        img = cv2.imread(scene.render.filepath + '.png', cv2.IMREAD_UNCHANGED)
        threshold = img[:, :, 3]
        mask = threshold != 0
        mask.dtype = np.uint8
        mask *= 255
        fm_path = os.path.join(fm, '%03d_%03d_%03d.png' % (n, int(eles[n]), int(azis[n])))
        cv2.imwrite(fm_path, mask)

        # resize and pad the original render to standard-sized render
        render_old = Image.open(scene.render.filepath + '.png').copy()
        bbox = render_old.getbbox()
        render_old = render_old.crop(bbox)
        render_new = resize_padding_v2(render_old, 224, 256)
        render_new.save(os.path.join(fi, '%03d_%03d_%03d.png' % (n, int(eles[n]), int(azis[n]))))        

        # add annotation for this sample
        cat_id = os.path.split(os.path.split(output_dir)[0])[1]
        example_id = os.path.split(output_dir)[1]
        image_path = os.path.join(output_dir, 'images', '%03d_%03d_%03d.png' % (n, int(eles[n]), int(azis[n])))
        with open(csv_file, 'a') as f:
            f.write(image_path + ',' + cat_id + ',' + example_id + ',' + str(int(azis[n])) + ',' + str(int(eles[n])) + '\n')


def render_obj(obj, output_dir, azi, ele, rol, name, shape=[512, 512], forward=None, up=None):
    clean_obj_lamp_and_mesh(bpy.context)

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
    lamp_object = makeLamp('Lamp1', 5)
    lamp_add = makeLamp('Lamp2', 1)

    if forward is not None and up is not None:
        bpy.ops.import_scene.obj(filepath=obj, axis_forward=forward, axis_up=up)
    else:
        bpy.ops.import_scene.obj(filepath=obj)

    # normalize it and set the center
    for object in bpy.context.scene.objects:
        if object.name in ['Camera', 'Lamp'] or object.type == 'EMPTY':
            continue
        bpy.context.scene.objects.active = object
        max_dim = max(object.dimensions)
        object.dimensions = object.dimensions / max_dim if max_dim != 0 else object.dimensions

    # Output setting
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = os.path.join(output_dir, name + '_rendering_%03d_%03d_%03d' % (int(azi), int(ele), int(rol)))

    # transform Euler angles from degrees into radians
    azi = radians(azi)
    ele = radians(ele)
    rol = radians(rol)
    r = 2.5
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


if __name__ == '__main__':
    obj = '/home/xiao/Projects/PoseFromShape/demo/armadillo.obj'
    render_dir = '/home/xiao/Projects/PoseFromShape/demo/armadillo_multiviews'
    render_obj_grid(obj, render_dir, [512, 512], 30, 5, 1, 2, True, None, None)
    
