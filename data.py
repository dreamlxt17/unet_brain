# -*- coding: utf-8 -*-
'''每次从所有slice中随机读取两张'''

import collections
import os
import random
import time
import warnings
from copy import deepcopy
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader
from glob import glob

class Brain_data(Dataset):
    def __init__(self, filelist, config, phase='train'):
        assert (phase == 'train' or phase == 'val' or phase == 'test' or
                phase == 'train+val')
        self.phase = phase

        self.blacklist = config['blacklist']
        self.isScale = config['aug_scale']
        self.r_rand = config['r_rand_crop']
        self.augtype = config['augtype']
        self.pad_value = config['pad_value']
        self.x_file_exp = config['x_file_exp']
        self.brainmask = config['x_brainmask']
        self.y_file_exp = config['y_file_exp']
        self.num_2d_per_img = config['num_2d_per_img']
        self.max_strides = config['max_strides']

        if phase == 'train' or phase == 'train+val':
            fileid_list = [os.path.split(filename)[1] for filename in filelist]
            black_files = [filelist[i] for i, f in enumerate(fileid_list)
                           if f in self.blacklist]
            for black_f in black_files:
                filelist.remove(black_f)

        self.x_filefolders = filelist

        self.crop = Crop(config)

    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  #seed according to time

        isRandomImg = False
        if self.phase != 'test':
            if idx >= len(self.x_filefolders):
                isRandom = True
                idx = idx % len(self.x_filefolders)
                isRandomImg = np.random.randint(2)
            else:
                isRandom = False
        else:
            isRandom = False

        if self.phase != 'test':
            if not isRandomImg:
                x_filefolder = self.x_filefolders[idx]
                x_data = []
                f_name = os.path.split(x_filefolder)[1]
                for f_exp in self.x_file_exp:
                    img_proxy = nib.load(os.path.join(x_filefolder, f_name +
                                                      f_exp))
                    channelData = img_proxy.get_data()
                    x_data.append(channelData)

                img_proxy = nib.load(os.path.join(x_filefolder,
                                                  self.brainmask))
                x_brainmask = img_proxy.get_data()

                img_proxy = nib.load(os.path.join(x_filefolder, f_name +
                                                  self.y_file_exp))
                y_data = img_proxy.get_data()

                isScale = self.augtype['scale'] and (self.phase == 'train' or
                                                     self.phase == 'train+val')

                sample, target, brainmask, coord = self.crop(
                    x_data, y_data, x_brainmask, isScale, isRandom)

                if self.phase == 'train' or self.phase == 'train+val' and not isRandom:
                    sample, target, brainmask, coord = augment(
                        sample,
                        target,
                        brainmask,
                        coord,
                        ifflip=self.augtype['flip'],
                        ifswap=self.augtype['swap'])
            else:
                randimid = np.random.randint(len(self.x_filefolders))

                x_filefolder = self.x_filefolders[randimid]
                x_data = []
                f_name = os.path.split(x_filefolder)[1]
                for f_exp in self.x_file_exp:
                    img_proxy = nib.load(os.path.join(x_filefolder, f_name +
                                                      f_exp))
                    channelData = img_proxy.get_data()
                    x_data.append(channelData)

                img_proxy = nib.load(os.path.join(x_filefolder,
                                                  self.brainmask))
                x_brainmask = img_proxy.get_data()

                img_proxy = nib.load(os.path.join(x_filefolder, f_name +
                                                  self.y_file_exp))
                y_data = img_proxy.get_data()

                sample, target, brainmask, coord = self.crop(x_data,
                                                             y_data,
                                                             x_brainmask,
                                                             isScale=False,
                                                             isRand=True)

            # assert()
            sample = (sample.astype(np.float32) - 128) / 128
            coord = np.tile(
                np.expand_dims(coord, 0), (self.num_2d_per_img, 1, 1, 1))

            return torch.from_numpy(sample), torch.from_numpy(target.astype(
                'int')), torch.from_numpy(brainmask.astype('int')), coord
        else:
            x_filefolder = self.x_filefolders[idx]
            x_data = []
            f_name = os.path.split(x_filefolder)[1]
            for f_exp in self.x_file_exp:
                img_proxy = nib.load(os.path.join(x_filefolder, f_name +
                                                  f_exp))
                channelData = img_proxy.get_data()
                x_data.append(channelData)

            img_proxy = nib.load(os.path.join(x_filefolder, self.brainmask))
            x_brainmask = img_proxy.get_data()

            sample = np.stack(x_data)
            sample = np.transpose(sample, (3, 0, 1, 2))
            brainmask = np.transpose(x_brainmask, (2, 0, 1))

            sample = (sample.astype(np.float32) - 128) / 128
            im_shape = sample.shape[-1]

            pad = [0, 0]
            if im_shape % 16 != 0:
                pad_v = self.max_strides - im_shape % 16
                pad = [pad_v / 2, pad_v - pad_v / 2]
                sample = np.pad(sample, [[0, 0], [0, 0], pad, pad],
                                'constant',
                                constant_values=self.pad_value)
                brainmask = np.pad(brainmask, [[0, 0], pad, pad],
                                   'constant',
                                   constant_values=self.pad_value)

            xx, yy = np.meshgrid(
                np.linspace(-0.5, 0.5, brainmask.shape[1]),
                np.linspace(-0.5, 0.5, brainmask.shape[2]),
                indexing='ij')
            coord = np.concatenate(
                [xx[np.newaxis, ...], yy[np.newaxis, ...]],
                0).astype('float32')

            coord = np.tile(
                np.expand_dims(coord, 0), (sample.shape[0], 1, 1, 1))

            return torch.from_numpy(sample), torch.from_numpy(
                brainmask), coord, np.array(pad)

    def __len__(self):
        if self.phase == 'train' or self.phase == 'train+val':
            return len(self.x_filefolders) / (1 - self.r_rand)
        elif self.phase == 'val':
            return len(self.x_filefolders)
        else:
            return len(self.x_filefolders)


def augment(sample, target, brainmask, coord, ifflip=True, ifswap=True):

    if ifswap:
        if sample.shape[1] == sample.shape[2]:
            axisorder = np.random.permutation(2)
            sample = np.transpose(sample,
                                  np.concatenate([[0], axisorder + 2, [3]]))
            coord = np.transpose(coord, np.concatenate([[0], axisorder + 1]))
            target = np.transpose(target, np.concatenate([[0], axisorder + 1]))
            brainmask = np.transpose(brainmask,
                                     np.concatenate([[0], axisorder + 1]))

    if ifflip:
        flip = np.random.randint(2) * 2 - 1
        sample = np.ascontiguousarray(sample[:, :, ::flip, :])
        coord = np.ascontiguousarray(coord[:, ::flip, :])
        target = np.ascontiguousarray(target[:, ::flip, :])
        brainmask = np.ascontiguousarray(brainmask[:, ::flip, :])

    return sample, target, brainmask, coord


class Crop(object):
    def __init__(self, config):
        self.crop_size = config['crop_size']
        self.bound_size = config['bound_size']
        self.pad_value = config['pad_value']
        self.num_2d_per_img = config['num_2d_per_img']

    def __call__(self,
                 x_data,
                 y_data,
                 x_brainmask,
                 isScale=False,
                 isRand=False):
        if isScale:
            scaleRange = [0.75, 1.25]

            scale = np.random.rand() * (scaleRange[1] -
                                        scaleRange[0]) + scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float') /
                         scale).astype('int')
        else:
            crop_size = self.crop_size

        # print(crop_size)
        bound_size = self.bound_size

        slice_num = x_brainmask.shape[-1]

        slice_list = []
        for i in range(slice_num):
            if np.unique(y_data[i]).size == 4:
                slice_list.append(i)

        if not slice_list:
            for i in range(slice_num):
                if np.unique(y_data[i]).size > 1:
                    slice_list.append(i)
        if not slice_list:
            raise ValueError('this person has no foreground labels')

        cared_2d_slice = np.random.choice(slice_list, self.num_2d_per_img) # 随机选８张有病变的slice

        cared_x_data = []
        for i in range(len(x_data)):
            cared_x_data.append(x_data[i][:, :, cared_2d_slice])
        cared_y_data = y_data[:, :, cared_2d_slice]
        cared_brainmask = x_brainmask[:, :, cared_2d_slice]

        start = []
        for i in range(2):
            if not isRand:
                s = np.floor((cared_brainmask.shape[i] - crop_size[i]) /
                             2.) + 1 - bound_size
                e = np.floor((cared_brainmask.shape[i] - crop_size[i]) /
                             2.) + 1 - bound_size

            else:
                s = np.floor((cared_brainmask.shape[i] - crop_size[i]) /
                             2.) + 1 - bound_size
                e = np.ceil(cared_brainmask.shape[i] - (cared_brainmask.shape[
                    i] - crop_size[i]) / 2. - bound_size - 1)

            if s > e:
                start.append(np.random.randint(e, s))  #!
            else:
                start.append(int(s + np.random.randint(-bound_size / 2,
                                                       bound_size / 2)))

        normstart = np.array(start).astype('float32') / np.array(
            cared_brainmask.shape[:2]) - 0.5
        normsize = np.array(crop_size).astype('float32') / np.array(
            cared_brainmask.shape[:2])

        xx, yy = np.meshgrid(
            np.linspace(normstart[0], normstart[0] + normsize[0],
                        self.crop_size[0]),
            np.linspace(normstart[1], normstart[1] + normsize[1],
                        self.crop_size[1]))

        coord = np.concatenate(
            [xx[np.newaxis, ...], yy[np.newaxis, ...]], 0).astype('float32')

        pad = []
        for i in range(2):
            leftpad = max(0, -start[i])
            rightpad = max(0,
                           start[i] + crop_size[i] - cared_brainmask.shape[i])
            pad.append([leftpad, rightpad])
        pad.append([0, 0])

        cared_x_crop_data = []
        for i in range(len(cared_x_data)):
            crop = cared_x_data[i][max(start[0], 0):min(start[0] + crop_size[
                0], cared_brainmask.shape[0]), max(start[1], 0):min(start[
                    1] + crop_size[1], cared_brainmask.shape[1]), :]
            # print(crop.shape)

            crop = np.pad(crop,
                          pad,
                          'constant',
                          constant_values=self.pad_value)
            # print(crop.shape)
            crop = np.transpose(crop, (2, 0, 1))
            cared_x_crop_data.append(crop)

        crop = cared_y_data[max(start[0], 0):min(start[0] + crop_size[
            0], cared_brainmask.shape[0]), max(start[1], 0):min(start[
                1] + crop_size[1], cared_brainmask.shape[1]), :]
        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)
        cared_y_crop_data = np.transpose(crop, (2, 0, 1))

        crop = cared_brainmask[max(start[0], 0):min(start[0] + crop_size[
            0], cared_brainmask.shape[0]), max(start[1], 0):min(start[
                1] + crop_size[1], cared_brainmask.shape[1]), :]
        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)
        cared_crop_brainmask = np.transpose(crop, (2, 0, 1))

        cared_xs = deepcopy(cared_x_crop_data)
        cared_y = deepcopy(cared_y_crop_data)
        cared_brainmask = deepcopy(cared_crop_brainmask)

        if isScale:
            cared_xs = []
            for i in range(len(cared_x_crop_data)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    crop = zoom(cared_x_crop_data[i], [1, scale, scale],
                                order=1)

                newpad = self.crop_size[0] - crop.shape[1:][0]
                if newpad < 0:
                    crop = crop[:, :-newpad, :-newpad]
                elif newpad > 0:
                    pad2 = [[0, 0], [0, newpad], [0, newpad]]
                    crop = np.pad(crop,
                                  pad2,
                                  'constant',
                                  constant_values=self.pad_value)
                cared_xs.append(crop)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cared_y = zoom(cared_y_crop_data, [1, scale, scale], order=1)
            newpad = self.crop_size[0] - cared_y.shape[1:][0]
            if newpad < 0:
                cared_y = cared_y[:, -newpad, :-newpad]
            elif newpad > 0:
                pad2 = [[0, 0], [0, newpad], [0, newpad]]
                cared_y = np.pad(cared_y,
                                 pad2,
                                 'constant',
                                 constant_values=self.pad_value)

            # print(cared_crop_brainmask.shape)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cared_brainmask = zoom(cared_crop_brainmask, [1, scale, scale],
                                       order=1)
            newpad = self.crop_size[0] - cared_brainmask.shape[1:][0]

            if newpad < 0:
                cared_brainmask = cared_brainmask[:, -newpad, :-newpad]
            elif newpad > 0:
                pad2 = [[0, 0], [0, newpad], [0, newpad]]

                cared_brainmask = np.pad(cared_brainmask,
                                         pad2,
                                         'constant',
                                         constant_values=self.pad_value)

        cared_xs = np.stack(cared_xs)
        cared_xs = np.transpose(cared_xs, (1, 0, 2, 3))
        return cared_xs, cared_y, cared_brainmask, coord


def collate(batch):
    if torch.is_tensor(batch[0]):
        return [b.unsqueeze(0) for b in batch]
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

if __name__=='__main__':
    from unet import config
    data_dir = '/all/Brats17TrainingData/preprocessed/'
    tr_filelist = glob(data_dir+'HGG/*') + glob(data_dir+'LGG/*')
    dataset = Brain_data(tr_filelist, config)
    data_loader = DataLoader(dataset)
    for i, data in enumerate(data_loader):
        print i

