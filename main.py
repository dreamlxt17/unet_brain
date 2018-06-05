# coding=utf-8
import random
import numpy as np
import torch
from torch.autograd import Variable
from glob import glob
from torch.utils.data import DataLoader
import tqdm

from loss import CrossEntropyLoss2d
from data import Brain_data
from unet import UnetSegment as Unet
from unet import ResBlock

EPOCH = 600
if_cuda = True
TRAIN = True
NC = 4
print_freq = 50

config = {}

config['crop_size'] = [208, 208]
config['n_classes'] = 4
config['colordim'] = 4
config['loss_name'] = 'cross_entropy'
config['class_weights'] = None
config['r_rand_crop'] = 0.3
config['pad_value'] = 0
config['bound_size'] = 12
config['max_strides'] = 16
config['aug_scale'] = True
config['augtype'] = {'flip': True, 'swap': False, 'scale': True}

config['x_file_exp'] = ['_flair.nii.gz', '_t1ce.nii.gz', '_t1.nii.gz',
                        '_t2.nii.gz']
config['x_brainmask'] = 'brainmask.nii.gz'
config['y_file_exp'] = '_seg.nii.gz'

config['num_2d_per_img'] = 8

config['blacklist'] = []


data_dir = '/all/Brats17TrainingData/preprocessed/'
tr_filelist = glob(data_dir+'HGG/*') + glob(data_dir+'LGG/*')
dataset = Brain_data(tr_filelist, config)
data_loader = DataLoader(dataset)
print len(dataset), len(tr_filelist)
unet = Unet(ResBlock, 4, num_classes=4)

def train():

    if if_cuda:
        torch.cuda.manual_seed(1)
        unet.cuda()

    criterion = CrossEntropyLoss2d()
    lr = 0.001
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    for e in range(EPOCH):
        save_e = random.randint(0,len(dataset))
        print save_e
        for i, (data, target, brainmask, coord) in enumerate(data_loader):
            data = data.view(-1, data.size(2), data.size(3), data.size(4))
            target = target.view(-1, target.size(2), target.size(3))
            brainmask = brainmask.view(-1, brainmask.size(2), brainmask.size(3))
            coord = coord.view(-1, coord.size(2), coord.size(3), coord.size(4))

            data = Variable(data.cuda(async=True))
            target = Variable(target.cuda(async=True))
            brainmask = Variable(brainmask.cuda(async=True))
            optimizer.zero_grad()
            outputs = unet(data)  # shape = [batch_size, num_class, 256, 256]

            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # print outputs.size()
            # print target.size()

            _, predicted = torch.max(outputs.data, 1)  # 此处参数选为1， 对第一维（num_class）操作

            if i % print_freq == 0:
                print ("Epoch [%d/%d], Iter [%d] Loss: %.4f" % (e + 1, EPOCH, i + 1, loss.data[0]))

            if i == save_e:
                np.savez('/home/didia/Didia/examples/brain/2d-unet/result/pred_label_{}_{}.npz'.format(e,i),
                         target.cpu().data.numpy(), Variable(predicted).cpu().data.numpy(),
                         brainmask.cpu().data.numpy(), data.cpu().data.numpy())
                print 'saved!_____________________________', e




train()