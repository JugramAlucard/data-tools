import os
import ipdb

def compare_and_delete_extra_files(folder_a, folder_b):

    # files_a = set(os.listdir(folder_a))
    # files_b = set(os.listdir(folder_b))
    file_list_a = [file_name[:-4] for file_name in os.listdir(folder_a)]
    file_list_b = [file_name[:-4] for file_name in os.listdir(folder_b)]
    
    # ipdb.set_trace()
    # 计算多余的文件
    extra_files = [file_name for file_name in file_list_a if file_name not in file_list_b]
    print(extra_files)
    # 删除多余的文件
    for extra_file in extra_files:
        file_path = os.path.join(folder_a, extra_file)
        os.remove(file_path + ".txt")
        print("Deleted {}".format(file_path))

# 用法示例
folder_a_path = r"D:\imageData\NJXWD\train\labels"
folder_b_path = r"D:\imageData\NJXWD\train\images"
compare_and_delete_extra_files(folder_a_path, folder_b_path)
