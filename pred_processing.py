# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import sys

result_dir = '/home/didia/Didia/examples/unet/brain/{}/'.format(sys.argv[1])
result = np.load(result_dir + 'pred_label_{}_{}.npz'.format(sys.argv[2], sys.argv[3]))  # 9, 11, 23,
# result = np.load(result_dir + 'pred_label_{}_{}.npz'.format(64, 0))  # 9, 11, 23,
data = result['arr_0']  # [8, 202, 208]
target = result['arr_1'] # [8, 208, 208]
out_put = result['arr_2'] # [8, 4, 208, 208]



# print target.shape, out_put.shape, data.shape
# print np.max(out_put)

# # for i in range(int(sys.argv[3])):
for i in range(8):

    print np.max(target[i]), '\t', np.max(out_put[i])

    fig, _ = plt.subplots()
    ax = plt.subplot(221)
    ax.set_title('Label')
    plt.imshow(target[i], cmap='gray' )
    ax = plt.subplot(222)
    ax.set_title('MRI')
    plt.imshow(data[i][0], cmap='gray')
    ax = plt.subplot(212)
    ax.set_title('Prediction')
    plt.imshow(out_put[i], cmap='gray')

    fig.suptitle('{}'.format(sys.argv[4]))

    plt.show()
