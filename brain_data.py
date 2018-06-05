# coding=utf-8
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from glob import glob
import torch
from scipy.ndimage import zoom
import nipy
import dicom

orgin_data_dir = '/all/DATA_PROCEING/total_original_data/'
roi_dir = '/all/DATA_PROCEING/total_ROI/'

class BrainData(Dataset):
    def __init__(self, filelist, shape=[208,208], scale_size=[0.85, 1.2], cared_slice = 4, train=True):
        super(BrainData, self).__init__()
        self.ann_list = filelist # 标注
        self.shape = shape
        self.scale_size = scale_size
        self.cared_slice = cared_slice
        self.train = train

        # 在validation阶段读取所有的dcm和nii数据
        if not self.train:
            self.total_data, self.total_roi, self.total_label = self.get_all_data()

    def get_all_data(self):

        total_data = []
        total_roi = []
        total_label = []
        w, h = self.shape
        for sample in self.ann_list:
            name = sample.split('/')[-1]
            nii = roi_dir + name + '_Merge.nii'
            label = int(sample.split('/')[-2])
            mask = np.transpose(nipy.load_image(nii), [2, 1, 0])
            size = mask.shape[-1]

            for i, ann in enumerate(mask):
                x = (dicom.read_file(sample + '/' + str(i + 1) + '.dcm')).pixel_array
                y = ann
                x, y = zoom(x, w*1.0 / size), zoom(y, h*1.0 / size)
                x, y = np.reshape(x, [1, w,h]), np.reshape(y, [1, w,h])
                total_data.append(x)
                total_roi.append(y)
                total_label.append(label)
        print len(total_data), len(total_roi)
        return total_data, total_roi, total_label


    def __getitem__(self, idx):
        if self.train:
            sample = self.ann_list[idx]
            name = sample.split('/')[-1]
            nii = roi_dir + name + '_Merge.nii'
            label = int(sample.split('/')[-2])
            mask = np.transpose(nipy.load_image(nii), [2,1,0])
            size = mask.shape[-1]

            x_data = []
            y_data = []
            slice_idxs = list(range(len(mask)))
            for i, ann in enumerate(mask):
                if np.max(ann)>0:
                    x_data.append((dicom.read_file(sample + '/' + str(i+1) + '.dcm')).pixel_array)
                    y_data.append(ann)
                    slice_idxs.remove(i)
            fine_slice = np.random.choice(slice_idxs, 4)
            for i in fine_slice:
                x_data.append((dicom.read_file(sample + '/' + str(i+1) + '.dcm')).pixel_array)
                y_data.append(mask[i])
            # print len(x_data)

            cared_index = np.random.choice(range(len(x_data)), self.cared_slice)
            cared_x = []
            cared_y = []
            for i in cared_index:
                x, y = x_data[i], y_data[i]
                x, y = zoom(x, 256.0/size), zoom(y, 256.0/size)  # 所有切片先resize成256x256大小, 然后再做scale或crop操作
                x, y = self.flip(x, y)
                x, y = self.random_scale(x, y)
                x, y = self.crop(x, y)
                cared_x.append(x)
                cared_y.append(y)

            return np.array(cared_x).astype('float'), np.array(cared_y).astype('float'), label*np.ones(self.cared_slice)
        else:
            return np.array(self.total_data[idx]).astype('float'), np.array(self.total_roi[idx]).astype('float'), self.total_label[idx]


    def random_scale(self, data, target):
        scale_size = self.scale_size
        scale = np.random.rand() * (scale_size[1] -
                                    scale_size[0]) + scale_size[0]
        data = zoom(data, scale)
        target = zoom(target, scale)

        return data, target

    def flip(self, data, target):
        flag = np.random.randint(0, 2)
        if flag:
            data = np.transpose(data, [1,0])
            target = np.transpose(target, [1,0])
        return data, target

    def crop(self, data, target):
        # 随机crop
        w, h = data.shape
        tw, th = self.shape
        if w == tw and h == th:
            return data, target
        # print w, tw
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        data = data[x1:x1 + tw, y1:y1 + th]
        target = target[x1:x1 + tw, y1:y1 + th]
        return np.reshape(data, [1,tw, th]), np.reshape(target, [1, tw, th])

    def __len__(self):
        if self.train:
            return len(self.ann_list)
        else:
            return len(self.total_data)

def get_slice(sample):
    dcm = sample.replace('_roi', '')
    x_data = np.load(dcm)
    for i, x in enumerate(x_data):
        x_data[i] = zoom(x, 208.0/512)
    return x_data

def get_pred_annotion(filelist):
    for i , sample in enumerate(filelist):
        x_data = get_slice(sample)


if __name__ == '__main__':

    all_file_list = glob(orgin_data_dir+'*/edema*')
    print len(all_file_list)

    dataset = BrainData(all_file_list, train=False)
    print(len(dataset))
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
    for i, (data, target, label) in enumerate(data_loader):
        print data.shape, target.shape


