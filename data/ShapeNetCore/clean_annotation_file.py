import pandas as pd
import os


data_frame = 'synthetic_annotation.txt'
df = pd.read_csv(data_frame)
print(len(df))

index = []
for i in range(len(df)):
    image_path = df.iloc[i, 0]
    if not os.path.exists(image_path):
        index.append(i)
        
df = df.drop(index, axis=0)
print(len(df))

df.to_csv('ShapeNetCore.txt', index=False)
