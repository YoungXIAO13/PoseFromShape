# test on the supervised categories
python testing.py --dataset ObjectNet3D --shape MultiView --model model/ObjectNet3D.pth  --output_dir ObjectNet3D

# test on the novel categories
python testing.py --dataset ObjectNet3D --shape MultiView --model model/ObjectNet3D_novel.pth --output_dir ObjectNet3D_novel
