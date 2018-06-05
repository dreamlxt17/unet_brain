# coding=utf-8
import random
import os
import numpy as np
import torch
from torch.autograd import Variable
from glob import glob
from torch.utils.data import DataLoader

from brain_data import BrainData
from loss import SegmentClassifyLoss, CrossEntropyLoss2d
from unet import UnetSegment as Unet
import warnings
warnings.filterwarnings("ignore")

EPOCH = 2000
if_cuda = True
TRAIN = True
LoadModel = False
print_freq =100
val_freq = 4
NC = 2 # 区分前景和背景
BN = 2 # batch_size
cared_slice = 4
input_size = [208, 208]
scale_size = [0.85, 1.2]
segment_weight = [0.15, 0.85]
classify_weight = [0.7, 0.3]
loss_weight = [1, 0]
train_dir = '/home/didia/Didia/examples/unet/brain/result/'
val_dir = '/home/didia/Didia/examples/unet/brain/val/'
model_dir = '/home/didia/Didia/examples/unet/brain/save_model_2/'

for d in [train_dir, val_dir, model_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

orgin_data_dir = '/all/DATA_PROCEING/total_original_data/'
roi_dir = '/all/DATA_PROCEING/total_ROI/'

unet = Unet(1, num_classes=NC)

def main(train_loader, val_loader, num_e=59):

    torch.cuda.manual_seed(1)
    unet.cuda()
    weight_1 = torch.from_numpy(np.array(segment_weight)).type(torch.FloatTensor).cuda()
    weight_2 = torch.from_numpy(np.array(classify_weight)).type(torch.FloatTensor).cuda()
    # criterion = SegmentClassifyLoss(weight_1, weight_2)
    criterion = CrossEntropyLoss2d(weight_1)
    lr = 0.001
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    if LoadModel:
        checkpoint = torch.load(model_dir + '{}.ckpt'.format(num_e))
        unet.load_state_dict(checkpoint['state_dict'])
        print 'Loading model~~~~~~~~~~', num_e

    for e in range(EPOCH):
        unet.train()
        class_correct = list(0. for i in range(NC))  # 1*2(number_of_classes)
        class_total = list(0. for i in range(NC))  # 1*2
        classify = []
        for i, (data, target, label) in enumerate(train_loader):
            # print data.shape
            data = data.type(torch.FloatTensor)
            data = data.view(-1, data.size(2), data.size(3), data.size(4)) # (BN*num_per_img, 1, shape[0], shape[1])
            data = Variable(data.cuda(async=True))

            target = target.view(-1, target.size(3), target.size(4)).type(torch.LongTensor) # (BN*num_per_img, shape[0], shape[1])
            target = Variable(target.cuda(async=True))

            label = label.view(-1).type(torch.LongTensor) # BN×1， 值为0或1
            label = Variable(label.cuda(async=True))

            optimizer.zero_grad()
            # outputs,pred_label = unet(data)  # shape = [batch_size, num_class, 256, 256], 预测的pixel-map和预测的label[BN, class]
            # loss1, loss2 = criterion(outputs, pred_label, target, label)
            # loss = loss1*loss_weight[0] + loss2*loss_weight[1]

            outputs= unet( data)  # shape = [batch_size, num_class, 256, 256], 预测的pixel-map和预测的label[BN, class]
            loss1 = criterion(outputs, target)
            loss = loss1

            # print loss1.data[0], loss2.data[0]
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)  # 此处参数选为1， 对第1维（num_class）操作
            correct = (predicted.cpu() == target.cpu().data).view(-1)
            tg = target.view(-1).cpu().data  # BATCH_SIZE*256*256, reshape成1xn的向量

            # _, classify_label = torch.max(pred_label.data, 1)
            # classify.append((classify_label.cpu()==label.cpu().data).numpy())

            for x, y in zip(tg, correct):
                class_correct[x] += y
                class_total[x] += 1

            if i % print_freq == 0:
                print ("Epoch [%d/%d], Iter [%d] Loss: %.4f" % (e, EPOCH, i, loss.data[0]))
                np.savez(train_dir + 'pred_label_{}_{}.npz'.format(e, i),
                data.cpu().data.numpy(),target.cpu().data.numpy(), Variable(predicted).cpu().data.numpy(),
                )
                # print 'training---------------:', e, i

        if e % val_freq==0:
            print '\t\t', 'recall', '\t\t', '\t', 'FPR','\t',  'classify accuracy'
            print class_correct[1]/class_total[1], '\t\t', 1-class_correct[0]/class_total[0]#,\
                                        #'\t\t' , np.sum(classify)*1.0/len(classify)/BN/cared_slice
            test(val_loader, e)

        if (e+1)%20 == 0:
            state_dict = unet.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            torch.save(
                {
                    'epoch': e,
                    'save_dir': model_dir,
                    'state_dict': state_dict,
                }, os.path.join(model_dir, '%d.ckpt' % e))

def test(val_loader, e=00, val=False):
    if val:
        unet.cuda()
        checkpoint =torch.load(model_dir + '1979.ckpt')
        unet.load_state_dict(checkpoint['state_dict'])
    unet.eval()

    class_correct = list(0. for i in range(NC))  # 1*3(number_of_classes)
    class_total = list(0. for i in range(NC))  # 1*3
    classify = []
    classify_correct = [0.0, 0.0]
    classify_total = [0.0, 0.0]
    save_random = np.random.randint(0, 10)

    for i, (data, target, label) in enumerate(val_loader):
        data = data.type(torch.FloatTensor)  # 数据类型转换
        # data = data.view(-1, data.size(2), data.size(3), data.size(4))  # (BN*num_per_img, 1, shape[0], shape[1])
        data = Variable(data.cuda(async=True))

        target = target.view(-1, target.size(3), target.size(4)).type(
            torch.LongTensor)  # (BN*num_per_img, shape[0], shape[1])

        label = label.view(-1).type(torch.LongTensor)  # BN×1， 值为0或1
        label = Variable(label.cuda(async=True))

        outputs, pred_label = unet(data)  # shape=[4, num_class, 256, 256]
        _, predicted = torch.max(outputs.data, 1)  # 此处参数选为1， 对第一维（num_class）操作
        predicted = predicted.cpu()

        _, classify_label = torch.max(pred_label.data, 1)
        classify.append((classify_label.cpu() == label.cpu().data).numpy())
        classify_temp = (classify_label.cpu() == label.cpu().data).view(-1)
        label = label.view(-1).data
        for x, y in zip(label, classify_temp):
            classify_correct[x] += y
            classify_total[x] += 1

        if i == save_random:  # 随机保存一个batch的数据
            image_data = data.data.cpu().numpy()
            label_data = Variable(target).data.numpy()
            predicted_data = Variable(predicted).cpu().data.numpy()
            np.savez(val_dir + 'pred_label_{}_{}.npz'.format(e, i), image_data, label_data, predicted_data)
            print "testing----------: %d" %e, i

        correct = (predicted.cpu() == target).view(-1)
        target = target.view(-1)  # BATCH_SIZE*256*256, reshape成1xn的向量

        for x, y in zip(target, correct):  # x可能是0,1,2(类别), y可能是0或1(表示正确或错误)
            class_correct[x] += y
            class_total[x] += 1

    print class_correct[1] / class_total[1], '\t\t', 1-class_correct[0]/class_total[0]#, \
    #     '\t\t', np.sum(classify) * 1.0 / len(classify) / BN / cared_slice, classify_correct[0]/classify_total[0], classify_correct[1]/classify_total[1]
    # return np.sum(classify) * 1.0 / len(classify) / BN / cared_slice, classify_correct[0]/classify_total[0], classify_correct[1]/classify_total[1]



if __name__ =='__main__':

    list_0 = glob(orgin_data_dir + '0/edema*')
    list_1 = glob(orgin_data_dir + '1/edema*')

    all_file_list = list_0+list_1
    print len(all_file_list)

    train_set = list_0[:17] + list_1[:29] + list_0[:17]
    val_set = list_0[17:] + list_1[29:]

    train_loader = DataLoader(BrainData(train_set, shape=input_size, cared_slice=cared_slice, scale_size=scale_size),
                              batch_size=BN, shuffle=True, num_workers=2)
    val_loader = DataLoader(BrainData(val_set, shape=input_size, cared_slice=cared_slice, train=False, scale_size=scale_size),
                            batch_size=BN, shuffle=True, num_workers=2)
    main(train_loader, val_loader, num_e=1979)





