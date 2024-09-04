import os
import shutil

src_dir = r'D:\imageData\NJXWD\real_ng'
dest_dir = r'D:\imageData\NJXWD\real_ng_result'
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
enhanced_img = []
result_img = []
for home, dir, files in os.walk(src_dir):
    for file_name in files:
        # print(files)
        enhanced_img.append(files[-2])
        result_img.append(files[-1])
# enhanced_elements = [sublist[-2] for sublist in file_list] # 增强
# result_elements = [sublist[-1] for sublist in file_list] # 结果 
# files = os.listdir(src_dir)   
    for file_name in result_img:
        full_file_name = os.path.join(home, file_name)
        
        if os.path.isfile(full_file_name):
            print(f"Copied: {full_file_name} to {dest_dir}")
            shutil.copy(full_file_name, dest_dir)
        


