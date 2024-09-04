import copy

import numpy as np
import os
import cv2
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import medfilt


def quintic_function(x, a, b, c, d, e, f, g):
    return a * x ** 6 + b * x ** 5 + c * x ** 4 + d * x ** 3 + e * x ** 2 + f * x + g
    # return  e * x ** 4 + f * x + g


def quadratic_function(x, a, b, c):
    return a * x ** 2 + b * x + c
    # return -a * np.exp(-(x-b)**2/(2*c**2)) + d


def tri_function(x, a, b, c, d, e, f):
    return f*x**4 + e*x**3 + a*x**2 + b*x + c + d/x


def nike_function(x, a, b, c):
    return a*x + b + np.exp(x)
    # return -a * np.exp(-(x-b)**2/(2*c**2)) + d

def gamma_correction(image_normalized, gamma):
    # 归一化到0-255范围
    # 应用伽马校正
    corrected_image = np.around(((image_normalized) ** gamma) * 255)
    return np.clip(corrected_image, 0, 255).astype(np.uint8)


def gmm_seg(matrix):
    data_flat = matrix.flatten().reshape(-1, 1)
    # 使用GMM拟合数据
    gmm = GaussianMixture(n_components=3, random_state=0)
    gmm.fit(data_flat)
    labels = gmm.predict(data_flat)
    means = gmm.means_.flatten()
    sorted_indices = np.argsort(means)
    # 获取两个高斯分布的均值
    mean1 = means[sorted_indices[0]]

    # extracted_pixels = np.where(extracted_pixels < mean1-1000, 0, matrix)
    return mean1


def find_largest_connected_region(raw_matrix):
    # 将输入矩阵转换为8位单通道图像
    matrix = raw_matrix.astype(np.uint8)

    # 找到所有连通区域
    contours, _ = cv2.findContours(matrix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果没有找到连通区域，返回原始矩阵
    if not contours:
        return matrix, None

    # 找到面积最大的连通区域
    max_contour = max(contours, key=cv2.contourArea)
    # 创建一个全零矩阵，大小与输入矩阵相同
    largest_region = np.zeros_like(raw_matrix)
    # 绘制面积最大的连通区域到全零矩阵
    cv2.drawContours(largest_region, [max_contour], -1, color=65536, thickness=cv2.FILLED)
    return largest_region, max_contour


def extract_bounding_square(matrix, mask, contour):
    # 计算连通区域的外接矩形
    x, y, w, h = cv2.boundingRect(contour)

    # 计算外接正方形的边长
    side_length = max(w, h)

    # 计算正方形的顶点坐标，使正方形包含外接矩形
    x1 = max(0, x + w // 2 - side_length // 2)
    y1 = max(0, y + h // 2 - side_length // 2)
    x2 = min(matrix.shape[1], x1 + side_length)
    y2 = min(matrix.shape[0], y1 + side_length)

    # 截取外接正方形内的矩阵
    square_matrix = matrix[y1:y2, x1:x2]
    square_mask = mask[y1:y2, x1:x2]

    return square_matrix, square_mask


def flatten_image_row(image_float32):
    res_image = copy.deepcopy(image_float32)
    for i, row in enumerate(res_image):
        line_non_zero = row[row > 0]
        if len(line_non_zero) < 800:
            row[row > 0] = 0
            continue
        x = np.arange(0, len(line_non_zero))
        popt, _ = curve_fit(quintic_function, x, line_non_zero, maxfev=10000)
        line_non_zero_flatten = line_non_zero - quintic_function(x, *popt)  #+ np.mean(line_non_zero)
        line_non_zero_flatten = medfilt(line_non_zero_flatten, 21)
        line_non_zero_flatten = medfilt(line_non_zero_flatten, 13)
        line_non_zero_flatten = medfilt(line_non_zero_flatten, 3)
        row[row > 0] = line_non_zero_flatten

        # 绘图
        # fig, ax1 = plt.subplots(figsize=(10, 6))
        #
        # ax1.plot(x, line_non_zero, label='Original Data')
        # ax1.plot(x, line_non_zero_flatten + np.mean(line_non_zero),
        #          label='Flattened Data')
        # ax1.plot(x, quintic_function(x, *popt), label='Fitted Curve')
        #
        # ax1.set_xlabel('Index')
        # ax1.set_ylabel('Value')
        # ax1.legend(loc='upper left')
        #
        # # 创建第二个 y 轴
        # ax2 = ax1.twinx()
        # ax2.plot(x, sigma, label='Sigma Array', linestyle='dashed', color='orange')
        # ax2.set_ylabel('Sigma Value')
        # ax2.legend(loc='upper right')
        #
        # plt.title('Curve Fitting with Sigma Array')
        # plt.show()
    return res_image


def flatten_image_columns(image_float32):
    res_image = copy.deepcopy(image_float32)
    for j in range(res_image.shape[1]):
        col = res_image[:, j]
        col_non_zero_edge = col[col > 0]
        if len(col_non_zero_edge) < 800:
            col[col > 0] = 0
            continue
        col_non_zero = col_non_zero_edge
        y = np.arange(-len(col_non_zero)-1, -1)
        popt, _ = curve_fit(tri_function, y, col_non_zero, maxfev=10000)
        col_non_zero_flatten = col_non_zero - tri_function(y, *popt)  #+ np.mean(col_non_zero)
        col_non_zero_flatten = medfilt(col_non_zero_flatten, 21)
        col_non_zero_flatten = medfilt(col_non_zero_flatten, 13)
        col_non_zero_flatten = medfilt(col_non_zero_flatten, 3)
        col[col > 0] = col_non_zero_flatten
        # plt.plot(col_non_zero, label='original')
        # plt.plot(col_non_zero_flatten + np.mean(col_non_zero), label='flatten')
        # plt.plot(tri_function(y, *popt), label='fit')
        # plt.legend()
        # plt.show()
    return res_image


def flatten_image_2d(image_float32):

    flatten_row = flatten_image_row(image_float32)
    flatten_col = flatten_image_columns(image_float32)
    non_zero_region = np.where((flatten_col != 0) & (flatten_row != 0))
    res = np.zeros_like(image_float32)
    res[non_zero_region] = (flatten_col[non_zero_region] + flatten_row[non_zero_region]) / 2
    # res = (flatten_col + flatten_row) / 2
    return res


def enhance_image(image):
    image1 = cv2.resize(image, (512, 512))
    threshold = gmm_seg(image1)
    image = np.where(image > threshold * 1.2, 0, image)
    preprocessed_matrix = np.copy(image)
    region_mask, max_contour = find_largest_connected_region(preprocessed_matrix)

    plt.imshow(region_mask)
    # plt.subplot(2,1,1)
    plt.show()

    # plt.imshow(max_contour)
    # # plt.subplot(2,1,2)
    # plt.show()

    if max_contour is not None:
        square_image, square_region_mask = extract_bounding_square(preprocessed_matrix, region_mask, max_contour)
        square_image = cv2.bitwise_and(square_image, square_region_mask)
        square_image = square_image.astype(np.float32)
        square_image = flatten_image_2d(square_image)

        square_image = np.clip(square_image, a_min=-127, a_max=127)
        # square_image[square_image < -125] = 0402
        # square_image[square_image > 125] = 0
        non_zero_region = np.where(square_image != 0)
        region_min = np.min(square_image[non_zero_region])
        region_max = np.max(square_image[non_zero_region])
        # print('min:', region_min, 'max:', region_max)

        # data_bin = square_image.flatten()
        # plt.figure(figsize=(10, 6))
        # plt.hist(data_bin, bins=30, edgecolor='black')
        # plt.title('Histogram of 2D Matrix Elements')
        # plt.xlabel('Value')
        # plt.ylabel('Frequency')
        # plt.grid(True)
        # plt.show()
        square_image[non_zero_region] = gamma_correction(
            (square_image[non_zero_region] - region_min) / (region_max - region_min), 1)
        square_image = square_image.astype(np.uint8)
        # square_image = cv2.medianBlur(square_image, 5)
        # square_image = cv2.equalizeHist(square_image)
        # square_image = cv2.fastNlMeansDenoising(square_image.astype(np.uint8), h=1, templateWindowSize=13, searchWindowSize=21)
        # square_image = cv2.medianBlur(square_image, 3)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(18, 18))
        # square_image = clahe.apply(square_image)

        # if file_i.replace('tif', 'png') in os.listdir('/Users/tony/Downloads/large_area_defect/images'):
        #     gt = cv2.imread('/Users/tony/Downloads/large_area_defect/images/' +
        #                     file_i.replace('raw', 'png'), cv2.IMREAD_GRAYSCALE)
        #     target_height, target_width = square_image.shape[:2]
        #     gt = cv2.resize(gt, [target_height, target_width])
        #     cv2.imshow('result_square_matrix8u',
        #                np.hstack([square_image.astype(np.uint8), gt.astype(np.uint8)]))
        #     cv2.imwrite('./0402_enhanced/' + file_i, np.hstack([square_image.astype(np.uint8), gt.astype(np.uint8)]))
        # else:
        # cv2.imshow('result_square_matrix8u', square_image.astype(np.uint8))
        # cv2.imwrite('./0402_enhanced/' + file_i, square_image.astype(np.uint8))

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return square_image.astype(np.uint8)
    return None


if __name__ == '__main__':
    for file_i in sorted(os.listdir('D:/imageData/0402/0402/')):
        if file_i.endswith('.tif'):
            print(file_i)
            image = cv2.imread('D:/imageData/0402/0402/' + file_i, cv2.IMREAD_UNCHANGED)
            # visualible_image = enhance_image(image)
            enhance_image(image)
            # cv2.imwrite('D:/imageData/original_image_train_enhanced/' + file_i, visualible_image)
    
    # for folder_i in ['N-160-3-39-juntu', '140-3.5-39-junt', '160-3-39-junt']:
    #     path_i = os.path.join('D:/imageData/original_image_test/', folder_i)
    #     for file_name in os.listdir(path_i):
    #         if file_name.endswith('.raw'):
    #             file_path = os.path.join(path_i, file_name) 
    #             data = np.fromfile(file_path, dtype=np.uint16).reshape(3072, 3072)
    #             image_enhanced = enhance_image(data)
    #             res_name = folder_i.replace('/', '_') + '_' + file_name.replace('.raw','.png')
    #             cv2.imwrite(os.path.join('D:/imageData/original_image_train_enhanced/', res_name), image_enhanced)
