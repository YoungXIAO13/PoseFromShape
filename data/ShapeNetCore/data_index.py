from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys

_cat_descs = {
  '02691156': 'airplane',
  '02773838': 'bag',
  '02801938': 'basket',
  '02808440': 'bathtub',
  '02818832': 'bed',
  '02828884': 'bench',
  '02843684': 'birdhouse',
  '02871439': 'bookshelf',
  '02876657': 'bottle',
  '02880940': 'bowl',
  '02924116': 'bus',
  '02933112': 'cabinet',
  '02747177': 'can',
  '02942699': 'camera',
  '02954340': 'cap',
  '02958343': 'car',
  '03001627': 'chair',
  '03046257': 'clock',
  '03207941': 'dishwasher',
  '03211117': 'display',
  '04379243': 'table',
  '04401088': 'telephone',
  '02946921': 'tin_can',
  '04460130': 'tower',
  '04468005': 'train',
  '03085013': 'keyboard',
  '03261776': 'earphone',
  '03325088': 'faucet',
  '03337140': 'file',
  '03467517': 'guitar',
  '03513137': 'helmet',
  '03593526': 'jar',
  '03624134': 'knife',
  '03636649': 'lamp',
  '03642806': 'laptop',
  '03691459': 'speaker',
  '03710193': 'mailbox',
  '03759954': 'microphone',
  '03761084': 'microwave',
  '03790512': 'motorcycle',
  '03797390': 'mug',
  '03928116': 'piano',
  '03938244': 'pillow',
  '03948459': 'pistol',
  '03991062': 'pot',
  '04004475': 'printer',
  '04074963': 'remote_control',
  '04090263': 'rifle',
  '04099429': 'rocket',
  '04225987': 'skateboard',
  '04256520': 'sofa',
  '04330267': 'stove',
  '04530566': 'watercraft',
  '04554684': 'washer',
  '02992529': 'cellphone'
}


def get_cat_ids():
    cat_ids = list(_cat_descs.keys())
    cat_ids.sort()
    return tuple(cat_ids)


_cat_ids = {v: k for k, v in _cat_descs.items()}


def cat_id_to_desc(cat_id):
    if isinstance(cat_id, (list, tuple)):
        return tuple(_cat_descs[c] for c in cat_id)
    else:
        return _cat_descs[cat_id]


def cat_desc_to_id(cat_desc):
    if isinstance(cat_desc, (list, tuple)):
        return tuple(_cat_ids[c] for c in cat_desc)
    else:
        return _cat_ids[cat_desc]


def to_cat_desc(cat):
    if cat in _cat_descs:
        return _cat_descs[cat]
    elif cat in _cat_ids:
        return cat
    else:
        raise ValueError('cat %s is not a valid id or descriptor' % cat)


def to_cat_id(cat):
    if cat in _cat_ids:
        return _cat_ids[cat]
    elif cat in _cat_descs:
        return cat
    else:
        raise ValueError('cat %s is not a valid id or descriptor' % cat)


_bad_objs = {'04090263': ('4a32519f44dc84aabafe26e2eb69ebf4',)}


def get_bad_objs(cat_id):
    """Get a tuple of ids that are "bad examples"""
    return _bad_objs.get(cat_id, ())


def get_example_ids(input_dir, cat_id, include_bad=False):
    data_path = os.path.join(input_dir, cat_id)
    if os.path.isdir(data_path):
        ids = [name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]
        ids.sort()
        if not include_bad:
            bad_objs = get_bad_objs(cat_id)
            ids = (i for i in ids if i not in bad_objs)
        return tuple(ids)
    else:
        raise ValueError('cat %s is not included' % cat_id)

