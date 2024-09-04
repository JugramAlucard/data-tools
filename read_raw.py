import numpy as np
import os
from enhance_exp1 import enhance_image
import cv2


for folder_i in ['N-160-3-39-juntu', '4th/140-3.5-39-junt', '4th/160-3-39-junt']:
    path_i = os.path.join('D:/imageData/original_image_test/dataset/', folder_i)
    for file_name in os.listdir(path_i):
        if file_name.endswith('.raw'):
            file_path = os.path.join(path_i, file_name)
            data = np.fromfile(file_path, dtype=np.uint16).reshape(3072, 3072)
            image_enhanced = enhance_image(data)
            res_name = folder_i.replace('/', '_') + '_' + file_name.replace('.raw','.png')
            cv2.imwrite(os.path.join('D:/imageData/original_image_train_enhanced/', res_name), image_enhanced)
