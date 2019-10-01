import os
from tqdm import tqdm


off_dir = "CAD/off"
obj_dir = "CAD/obj"

cats = [name for name in os.listdir(off_dir)]
cats.sort()

for cat in tqdm(cats):
    input_dir = os.path.join(off_dir, cat)
    output_dir = os.path.join(obj_dir, cat)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    files = [name for name in os.listdir(input_dir) if name.endswith(".off") and len(name.split(".")[0]) == 2]
    files.sort()

    for f in files:
        os.system("meshlabserver -i %s -o %s" % (os.path.join(input_dir, f), os.path.join(output_dir, f.replace(".off", ".obj"))))
    
