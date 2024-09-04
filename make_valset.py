import os
import random
import shutil

TRAIN_PER = 0.8
OUTER_PER = 0.2

img_train_root = r'D:\imageData\NJXWD\train\images'
label_train_root = r'D:\imageData\NJXWD\train\labels'
img_destination_folder = r'D:\imageData\NJXWD\val\images'
label_destination_folder = r'D:\imageData\NJXWD\val\labels'
# img_outer_root = ''

train_path = os.listdir(img_train_root)
# outer_path = os.listdir(img_outer_root)

train_num = len(train_path) 
train_list = list(range(train_num))
train_add_list = random.sample(train_list, int(train_num * TRAIN_PER))

for i in train_add_list:
    name = train_path[i][:-4]
    print(name)
    img_src_path = os.path.join(img_train_root, name + '.png')
    img_dst_path = os.path.join(img_destination_folder, name + '.png')
    label_src_path = os.path.join(label_train_root, name + '.txt')
    label_dst_path = os.path.join(label_destination_folder, name + '.txt')
    shutil.copy(img_src_path, img_dst_path)
    shutil.copy(label_src_path, label_dst_path)
