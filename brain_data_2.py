# coding=utf-8
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from glob import glob
from scipy.ndimage import zoom


from augmentor import center_crop, random_scale, random_crop, elastic_transform, random_rotated
'''一次性load所有切片'''


class BrainData(Dataset):
    def __init__(self, filelist, shape=[208,208], scale_size=[0.85, 1.2], cared_slice = 1, train=True):
        super(BrainData, self).__init__()
        self.ann_list = filelist # 标注
        self.shape = shape
        self.scale_size = scale_size
        self.cared_slice = cared_slice
        self.train = train

        slices = []
        # 获取所有的slice对应的dcm与target, 以list形式存储
        for i, sample in enumerate(self.ann_list):
            dcm = sample.replace('_roi', '')
            label = int(sample.split('/')[-2])
            x_data = np.load(dcm)
            y_target = np.load(sample)
            for d, a in zip(x_data, y_target):
                slices.append([d, a, label])
        self.slices = slices

    def __getitem__(self, idx):
        x, y, label = self.slices[idx]
        x = zoom(center_crop(x, [208, 208]), 256.0/208) # 256x256大小的图片
        y = zoom(center_crop(y, [208, 208]), 256.0/208)
        w, h = self.shape
        if self.train:
            x, y = random_scale(x, y)
            x, y = elastic_transform(x, y)
            x, y = random_rotated(x, y)
            x, y = random_crop(x, y)
        else:
            x, y = zoom(x, w*1.0/256), zoom(y, h*1.0/256)
            x, y = np.reshape(x, [1, w, h]), np.reshape(y, [1, w, h])

        return np.array(x).astype('float'), np.array(y).astype('float'), label  # 208x208大小

    def __len__(self):
        return len(self.slices)

if __name__ == '__main__':

    # segment_dir = '/all/DATA_PROCEING/segment/' # 所有包括病变位置的切片
    segment_dir = '/all/DATA_PROCEING/segment_all/' # 所有
    all_file_list = glob(segment_dir + '*/edema*_roi.npy')  # 所有的annotation 按人计算
    print len(all_file_list)


    dataset = BrainData(all_file_list, shape=[208,208])
    print len(dataset)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
    for i, (data, target, label) in enumerate(data_loader):
        print data.shape, target.shape
        plt.imshow(data[0][0])
        plt.show()
        break


