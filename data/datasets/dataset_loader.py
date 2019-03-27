# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
import os.path as osp

from torch.utils.data import Dataset


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


def read_image(img_path, camid="0"):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    image_path = img_path[0]
    box = img_path[1]

    if not osp.exists(image_path):
        raise IOError("{} does not exist".format(image_path))
    while not got_img:
        try:
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8()), 1)
            box = process_box(box, img.shape[1], img.shape[0])
            img = img[box[1]:box[3], box[0]:box[2], :]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(144, 144))
            if camid:
                img = cv2.flip(img, -1)
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path[0]
