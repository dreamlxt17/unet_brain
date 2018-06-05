# coding=utf-8

import numpy as np
from scipy.ndimage import zoom
from skimage.util import random_noise
import random
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates, rotate
import matplotlib.pyplot as plt

def center_crop(data, crop_size=[208, 208]):
    w, h = data.shape
    pw = (w - crop_size[0]) / 2
    ph = (h - crop_size[1]) / 2
    data = data[pw:crop_size[0] + pw, ph:crop_size[1] + ph]
    return data

def random_scale(data, target, scale_size=[0.85, 1.2]):

    scale = np.random.rand() * (scale_size[1] -
                                scale_size[0]) + scale_size[0]
    data = zoom(data, scale)
    target = zoom(target, scale)

    return data, target

def random_flip(data, target):
    flag = np.random.randint(0, 2)
    if flag:
        data = np.transpose(data, [1,0])
        target = np.transpose(target, [1,0])
    return data, target

def random_rotated(data, target):
    angle_range = [0.0, 360.0]
    angle = np.random.uniform(angle_range[0], angle_range[1])
    x = rotate(data, angle)
    y = rotate(target, angle)
    return x, y

def random_crop(data, target, shape=[208, 208]):
    # 随机crop
    w, h = data.shape
    tw, th = shape
    if w == tw and h == th:
        return data, target
    # print w, tw
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    data = data[x1:x1 + tw, y1:y1 + th]
    target = target[x1:x1 + tw, y1:y1 + th]
    return np.reshape(data, [1,tw, th]), np.reshape(target, [1, tw, th])

def noise(data):
    flag = np.random.randint(0, 2)
    if flag:
        data = random_noise(data)
    return data

def random_num_generator(config, random_state=np.random):
    # 产生随机数，config = [随机函数类型， start， end]
    if config[0] == 'uniform':
        ret = random_state.uniform(config[1], config[2], 1)[0]
    elif config[0] == 'lognormal':
        ret = random_state.lognormal(config[1], config[2], 1)[0]
    else:
        print(config)
        raise Exception('unsupported format')
    return ret

def elastic_transform(data, target, alpha=1000, sigma=30, spline_order=1, mode='nearest', random_state=np.random):
    shape = data.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
    x = map_coordinates(data, indices, order=spline_order, mode=mode).reshape(shape)
    y = map_coordinates(target, indices, order=spline_order, mode=mode).reshape(shape)

    # plt.subplot(221)
    # plt.imshow(image)
    # plt.subplot(222)
    # plt.imshow(target)
    # plt.subplot(223)
    # plt.imshow(x)
    # plt.subplot(224)
    # plt.imshow(y)
    # plt.show()

    return x, y

