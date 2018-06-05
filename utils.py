#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: utils.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017-06-20 14:56:54

**Note.** This code absorb some code from following source.
1. [DSB2017](https://github.com/lfz/DSB2017)
"""

import os
import sys

import numpy as np
import torch
from random import shuffle

def split_fold(len_data, n=5):
    idxs = range(len_data)
    shuffle(idxs)
    split = len(idxs)/n
    idxs1 = idxs[:split]
    idxs2 = idxs[split:2*split]
    idxs3 = idxs[2*split:3*split]
    idxs4 = idxs[3*split:4*split]
    idxs5 = idxs[4*split:]

    tr1, te1 = idxs2+idxs3+idxs4+idxs5, idxs1
    tr2, te2 = idxs1+idxs3+idxs4+idxs5, idxs2
    tr3, te3 = idxs2+idxs1+idxs4+idxs5, idxs3
    tr4, te4 = idxs2+idxs3+idxs1+idxs5, idxs4
    tr5, te5 = idxs2+idxs3+idxs4+idxs1, idxs5

    return [[tr1, te1], [tr2, te2], [tr3, te3], [tr4, te4], [tr5, te5],]



def getFreeId():
    import pynvml

    pynvml.nvmlInit()

    def getFreeRatio(id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        use = pynvml.nvmlDeviceGetUtilizationRates(handle)
        ratio = 0.5 * (float(use.gpu + float(use.memory)))
        return ratio

    deviceCount = pynvml.nvmlDeviceGetCount()
    available = []
    for i in range(deviceCount):
        if getFreeRatio(i) < 70:
            available.append(i)
    gpus = ''
    for g in available:
        gpus = gpus + str(g) + ','
    gpus = gpus[:-1]
    return gpus


def setgpu(gpuinput):
    freeids = getFreeId()
    if gpuinput == 'all':
        gpus = freeids
    else:
        gpus = gpuinput
        busy_gpu = [g not in freeids for g in gpus.split(',')]
        if any(busy_gpu):
            raise ValueError('gpu' + ' '.join(busy_gpu) + 'is being used')
    print('using gpu ' + gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    return len(gpus.split(','))


class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass
