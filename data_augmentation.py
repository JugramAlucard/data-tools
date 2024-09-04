import torch
import os
from torchvision import transforms
from PIL import Image
import cv2 as cv

img_path = "D:/imageData/NJXWD/20240708/train/images/"

for img_name in os.listdir(img_path):

    flip_transformer = transforms.Compose([
            transforms.RandomHorizontalFlip(),  
            transforms.RandomVerticalFlip()  
        ])
    # img = cv.imread(img_path + img_name)
    img = Image.open(img_path + img_name)
    flip_img = flip_transformer(img)
    # cv.imwrite(flip_img, f"{img_path} + 'trans_' + {img_name}")
    flip_img.save(img_path + 'trans_' + img_name, 'png')