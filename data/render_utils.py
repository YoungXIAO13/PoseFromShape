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


def render_obj_grid(obj, output_dir, shape=[256, 256], step=30, light_main=5, light_add=1, r=1.5, normalize=False, offset=0, forward=None, up=None):
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
    lamp_object = makeLamp(light_main)
    lamp_add = makeLamp(light_add)

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
    
        loc_y = r * math.cos(radians(ele)) * math.cos(radians(azi + offset))
        loc_x = r * math.cos(radians(ele)) * math.sin(radians(azi + offset))
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


if __name__ == '__main__':
    obj = 'CAD/obj/aeroplane/01.obj'
    render_dir = 'Renders_semi_sphere'
    shape = [512, 512]
    render_obj_grid(obj, render_dir, shape, 45)
    

