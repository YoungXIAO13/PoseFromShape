import os
from tqdm import tqdm

ply_dir = "models"
obj_dir = "models"

plys = [name for name in os.listdir(ply_dir) if name.endswith(".ply")]

for ply in tqdm(plys):
    os.system("meshlabserver -i %s -o %s" % (os.path.join(ply_dir, ply), os.path.join(obj_dir, ply.replace(".ply", ".obj")[4:])))
