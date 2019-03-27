# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing, Random2DTranslation, RandomFlip, ColorAugmentation
from .opencv_transforms import ColorJitter as ColorJitter_cv
from .opencv_transforms import Resize as Resize_cv
from .opencv_transforms import RandomCrop as RandomCrop_cv
from .opencv_transforms import Pad as Pad_cv


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            Resize_cv(tuple(cfg.INPUT.SIZE_TRAIN)),
            RandomCrop_cv(tuple(cfg.INPUT.SIZE_TRAIN)),
            Pad_cv(cfg.INPUT.PADDING),
            ColorJitter_cv(brightness=0.2, contrast=0.15, saturation=0, hue=0),
            T.ToTensor(),
            RandomFlip(probability=cfg.INPUT.PROB, dim=1),
            RandomFlip(probability=cfg.INPUT.PROB, dim=2),
            ColorAugmentation(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
