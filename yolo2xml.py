import os
import cv2
from xml.dom.minidom import Document

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(txt_path, xml_path, img_path):
    classes = ["stain", "fold_corner", "shadow", "fold", "broken", "crack"]  # 类别名称列表
    txt_files = os.listdir(txt_path)
    for txt_file in txt_files:
        img_file = txt_file[:-4] + '.tiff'
        print(img_file)
        img = cv2.imread(os.path.join(img_path, img_file))
        height, width, depth = img.shape

        doc = Document()
        annotation = doc.createElement('annotation')
        doc.appendChild(annotation)

        folder = doc.createElement('folder')
        folder.appendChild(doc.createTextNode('JPEGImages'))
        annotation.appendChild(folder)

        filename = doc.createElement('filename')
        filename.appendChild(doc.createTextNode(str(img_file)))
        annotation.appendChild(filename)

        path = doc.createElement('path')
        path.appendChild(doc.createTextNode(str(img_path + img_file)))
        annotation.appendChild(path)

        size = doc.createElement('size')
        annotation.appendChild(size)

        width_node = doc.createElement('width')
        width_node.appendChild(doc.createTextNode(str(width)))
        size.appendChild(width_node)

        height_node = doc.createElement('height')
        height_node.appendChild(doc.createTextNode(str(height)))
        size.appendChild(height_node)

        depth_node = doc.createElement('depth')
        depth_node.appendChild(doc.createTextNode(str(depth)))
        size.appendChild(depth_node)

        with open(os.path.join(txt_path, txt_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_id, x_center, y_center, w, h = map(float, line.split())
                x_min = (x_center - w / 2) * width
                y_min = (y_center - h / 2) * height
                x_max = (x_center + w / 2) * width
                y_max = (y_center + h / 2) * height

                object = doc.createElement('object')
                annotation.appendChild(object)

                name = doc.createElement('name')
                name.appendChild(doc.createTextNode(classes[int(class_id)]))
                object.appendChild(name)

                difficult = doc.createElement('difficult')
                difficult.appendChild(doc.createTextNode(str(0)))
                object.appendChild(difficult)

                bndbox = doc.createElement('bndbox')
                object.appendChild(bndbox)

                xmin = doc.createElement('xmin')
                xmin.appendChild(doc.createTextNode(str(int(x_min))))
                bndbox.appendChild(xmin)

                ymin = doc.createElement('ymin')
                ymin.appendChild(doc.createTextNode(str(int(y_min))))
                bndbox.appendChild(ymin)

                xmax = doc.createElement('xmax')
                xmax.appendChild(doc.createTextNode(str(int(x_max))))
                bndbox.appendChild(xmax)

                ymax = doc.createElement('ymax')
                ymax.appendChild(doc.createTextNode(str(int(y_max))))
                bndbox.appendChild(ymax)

        xml_str = doc.toprettyxml(indent="  ")
        save_path = os.path.join(xml_path, txt_file[:-4] + '.xml')
        with open(save_path, 'w') as f:
            f.write(xml_str)

# 调用函数
txt_path = r'D:\imageData\fasterrcnn_dataset\labels'  # txt文件所在路径
xml_path = r'D:\anaconda3\envs\pytorch\simple-faster-rcnn-pytorch-master\dataset\PascalVOC\VOC2007\Annotations'  # xml文件保存路径
img_path = 'D:/anaconda3/envs/pytorch/simple-faster-rcnn-pytorch-master/dataset/PascalVOC/VOC2007/JPEGImages/'     # 图片文件所在路径
convert_annotation(txt_path, xml_path, img_path)
