import os
import pandas as pd

sort_img_path = r'D:\imageData\all_images' 
junt_img_path = r'D:\imageData\dazhong_source'

flag_dict = {}
for img_home, img_dir, img_name in os.walk(sort_img_path):
        for filename in img_name:
            flag_dict[filename] = img_home.split('\\')[-1]
            print(img_home)

junt_dic = {}
for junt_home, junt_dir, junt_name in os.walk(junt_img_path):
    for filename in junt_name:
        junt_dic[filename] = junt_home.split('\\')[-1]
    
flag_list = []
name_list =[]
box_list = []
for key in flag_dict:
     name_list.append(key)
     flag_list.append(flag_dict.get(key))
     box_list.append(junt_dic.get(key))
    
dataframe = pd.DataFrame({'名称':name_list, '判别':flag_list, '箱号':box_list}, columns=['名称','判别','箱号'])
dataframe.to_csv(r"D:\imageData\test.csv", index=False, sep=',')



