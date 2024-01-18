import os
import random
import shutil
import time

source_img = 'remain_img3'
source_anno = 'remain_anno3'
#train_img = 'train_remain_images'
#train_anno = 'train_remain_annotations'
val_img = 'val_remain_img3'
val_anno = 'val_remain_anno3'
txt_files = os.listdir(source_anno)
num_files_val = round(len(txt_files) * 0.15) 
#num_files_train = len(txt_files) - num_files_val

print(num_files_val)
print(os.path.isdir(source_img))
print(os.path.isdir(source_anno))
#print(os.path.isdir(train_img))
#print(os.path.isdir(train_anno))
print(os.path.isdir(val_img))
print(os.path.isdir(val_anno))


for file_val in random.sample(txt_files, num_files_val):
    shutil.move(os.path.join(source_anno, file_val), val_anno)
    png_val = str(file_val.split(".txt")[0]) + ".jpg"
    shutil.move(os.path.join(source_img, png_val), val_img)
