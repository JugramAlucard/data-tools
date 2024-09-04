import os

CLASS_NAME = {
    0: "stain",
    1: "fold_corner",
    2: "shadow",
    3: "fold",
    4: "broken",
    5: "crack",
    6: "bar",
    7: "scratch",
    8: "blackhole",
    9: "particle"
}

def count_class_samples(txt_path, class_num):
    
    class_num_list = [0] * class_num
    labels_list = os.listdir(txt_path)
    labels_list.remove('classes.txt')

    for label_file in labels_list:
        file_path = os.path.join(txt_path, label_file)
        with open(file_path, 'r') as file:
            file_data = file.readlines()
            for every_row in file_data:
                class_val = int(every_row.split(' ')[0])
                class_num_list[class_val] += 1
                
                if class_val == 7:    
                    print(label_file) # To find certain label(0 as stain...)

    for class_idx, count in enumerate(class_num_list):
        print(f"{CLASS_NAME.get(class_idx)}: {count}")
    print("Total samples:", sum(class_num_list))

if __name__ == '__main__':
    
    txt_folder_path = r'D:\\imageData\\NJXWD\\train\\labels\\'
    num_classes = 10 # 样本类别数
    count_class_samples(txt_folder_path, num_classes)
