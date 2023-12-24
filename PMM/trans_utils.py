import numpy as np
import random

import torch
from torchvision import transforms

from constants import *
from angle_utils import rot_aa


totensor = transforms.ToTensor()
bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.98]))


def get_random_value(lower, upper):
    return np.random.uniform()*(lower - upper) + upper


def RandomAffine(images, pmap, degrees=0, scale=0, translate=0.15, p=0.7, separate=False):
    if np.random.uniform() < (1-p):
        return images, pmap

    def get_params():
        # return (get_random_value(-degrees, degrees), \
        #         get_random_value(-translate, translate), \
        #         get_random_value(-translate, translate), \
        #         get_random_value(-scale, scale) + 1)
        return (get_random_value(-translate, translate), get_random_value(-translate, translate))

    trans_factor_x, trans_factor_y = get_params()
    out_images = []
    for image in images:
        if separate:
            trans_factor_x, trans_factor_y = get_params()
        out_images.append(transforms.functional.affine(image, angle=0, \
                        translate=[trans_factor_x*image.shape[2], trans_factor_y*image.shape[1]], \
                        scale=1, shear=0))
    return out_images, pmap


def RandomCutOut(images, pmap, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0, p=0.6, num_cuts=5): #earlier scale was (0.02, 0.33)
    transform = transforms.RandomErasing(p=p, scale=scale, ratio=ratio)
    out_images = []
    for image in images:
        for _ in range(num_cuts):
            image = transform(image)
        out_images.append(image)
    return out_images, pmap


def Rotate(pose, pressure_image, depth_image, deg=15., p=0.9):
    if np.random.uniform() < (1-p):
        return pose, pressure_image, depth_image
    rot = np.clip(np.random.randn(), -2., 2.)*deg
    pose[:3] = rot_aa(pose[:3], -rot)
    pressure_image = transforms.functional.rotate(pressure_image, rot)
    depth_image = transforms.functional.rotate(depth_image, rot)
    return pose, pressure_image, depth_image


def PixelDropout(images, pmap):
    out_images = []
    for image in images:
        noise = bernoulli.sample(image.shape).squeeze(-1)
        image *= noise 
        out_images.append(image)
    return out_images, pmap


def ToTensor(images, pmap):
    out_images = []
    for image in images:
        out_images.append(totensor(image))
    return out_images, pmap


def Resize(images, pmap, size=(224, 224)):
    transform = transforms.Resize(size, antialias=None)
    out_images = []
    for image in images:
        out_images.append(transform(image))
    return out_images, pmap

