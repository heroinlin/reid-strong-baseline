# -*- coding: utf-8 -*-

import os
import glob
import cv2
import math
import torchvision.transforms as tv_transforms
from runstats import Statistics
import numpy as np
import json
import pickle


def process_box(box, image_width, image_height):
    x1, y1, x2, y2 = box
    x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    # 框扩大1.5倍
    w = min(w * 1.5, 1.0)
    h = min(h * 1.5, 1.0)
    x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    # 到图像范围
    x1, y1, x2, y2 = round(x1 * image_width), round(y1 * image_height), round(x2 * image_width), round(
        y2 * image_height)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, image_width), min(y2, image_height)
    box = [x1, y1, x2, y2]
    return box


def read_image(image_path, box):
    box = process_box(box, 320, 288)
    image = cv2.imread(image_path, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[box[1]:box[3], box[0]:box[2], :]
    image = cv2.resize(image, dsize=(144, 144))
    image = tv_transforms.ToTensor()(image)
    image = image.numpy()
    return image


def compute_mean_and_stddev(image_root, pickle_path):
    r_mean, g_mean, b_mean = Statistics(), Statistics(), Statistics()
    r_square_mean, g_square_mean, b_square_mean = Statistics(), Statistics(), Statistics()
    count = 1
    annotations_file = open(pickle_path, "rb")
    annotations = pickle.load(annotations_file)
    annotations_file.close()
    for annotation in annotations:
        image_path = os.path.join(image_root, annotation["image"]["information"]["path"])
        box = annotation["annotation"]["persons"][0]["box"]
        image = read_image(image_path, box)
        r_channel = image[0].flatten().tolist()
        g_channel = image[1].flatten().tolist()
        b_channel = image[2].flatten().tolist()
        r_s, g_s, b_s = Statistics(), Statistics(), Statistics()
        r_square_s, g_square_s, b_square_s = Statistics(), Statistics(), Statistics()
        for e in r_channel:
            r_s.push(e)
            r_square_s.push(e**2)
        for e in g_channel:
            g_s.push(e)
            g_square_s.push(e**2)
        for e in b_channel:
            b_s.push(e)
            b_square_s.push(e**2)
        r_mean.push(r_s.mean())
        g_mean.push(g_s.mean())
        b_mean.push(b_s.mean())
        r_square_mean.push(r_square_s.mean())
        g_square_mean.push(g_square_s.mean())
        b_square_mean.push(b_square_s.mean())
        if count % 100 == 0:
            print('processed process: {:05d}/{}'.format(count, len(annotations)))
        count += 1
    mean = np.array([r_mean.mean(), g_mean.mean(), b_mean.mean()])
    stddev = np.array([math.sqrt(r_square_mean.mean() - mean[0] ** 2),
                       math.sqrt(g_square_mean.mean() - mean[1] ** 2),
                       math.sqrt(b_square_mean.mean() - mean[2] ** 2)])
    return mean, stddev, len(annotations)


def merge_mean_stddev(n, mean1, stddev1, m, mean2, stddev2):
    """
    已知两组数据的个数，均值和方差，求总数据的均值和标准差
    Args:
        n: 第一组数据的个数
        mean1: 第一组数据的均值
        stddev1: 第一组数据的标准差
        m: 第二组数据的个数
        mean2: 第二组数据的均值
        stddev12: 第二组数据的标准差

    Returns:
        所有数据的个数，均值，标准差
    """
    mean = (n * mean1 + m * mean2) / (m + n)
    var = (n*(stddev1**2 + mean1**2) + m*(stddev2**2 + mean2**2))/(m+n) - mean**2
    return m+n, mean, np.sqrt(var)


def merge_mean_stddev_test():
    image_root = r"F:\Database\od_dataset\train_format"
    train_1_pkl = r"F:\Database\od_dataset\partition_file\train\train_1.pkl"
    train_2_pkl = r"F:\Database\od_dataset\partition_file\train\train_2.pkl"
    mean1, stddev1, len1 = compute_mean_and_stddev(image_root, train_1_pkl)
    mean2, stddev2, len2 = compute_mean_and_stddev(image_root, train_2_pkl)
    print(merge_mean_stddev(len1, mean1, stddev1, len2, mean2, stddev2))


def merge_mean_stddev_folder(mean_stddev_folde):
    length, mean, stddev = 0, 0, 0
    for mean_var_json in glob.glob(mean_stddev_folde + "/*.json"):
        json_file = open(mean_var_json, "r", encoding='utf-8')
        information = json.load(json_file)
        length1 = information["count"]
        mean1 = np.array(information["mean"])
        stddev1 = np.array(information["stddev"])
        json_file.close()
        length, mean, stddev = merge_mean_stddev(length, mean, stddev, length1, mean1, stddev1)
    print(length, mean, stddev)
    json_file = open(mean_stddev_folde + "/total_mean_stddev.json", "w", encoding='utf-8')
    json.dump({"count": length, "mean": mean.tolist(), "stddev": stddev.tolist()}, json_file, indent=4)
    json_file.close()


def main():
    image_root = r"F:\Database\od_dataset\train_format"
    train_pkl = r"F:\Database\od_dataset\partition_file\train\train_3.pkl"
    mean, stddev, length = compute_mean_and_stddev(image_root, train_pkl)
    print('mean: {}'.format(mean))      # mean:   (0.4461771089370513, 0.446075284259276, 0.44311939505836345)
    print('stddev: {}'.format(stddev))  # stddev: (0.2480288227742099, 0.24619806742784317, 0.24880323463000772)
    print(length)
    json_file = open(train_pkl.replace('.pkl', '.json'), "w", encoding='utf-8')
    json.dump({"count": length, "mean": mean.tolist(), "stddev": stddev.tolist()}, json_file, indent=4)
    json_file.close()


if __name__ == "__main__":
    main()
    # merge_mean_stddev_test()
    # merge_mean_stddev_folder(r"F:\Database\od_dataset\partition_file\train")

