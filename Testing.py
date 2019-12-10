#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:09:36 2019

@author: haici
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


import torch, gzip
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import _pickle as pickle
from torch.utils.data import TensorDataset, DataLoader

dtype=torch.cuda.FloatTensor
import matplotlib.pyplot as plt

import librosa
import librosa.display

import torch.nn.utils.rnn as rnn
import time
import numpy as np
#%matplotlib notebook



def SDR(s,sr): # s - original input; sr --  cleaned one
    eps=1e-20
    ml=np.minimum(len(s), len(sr))
    s=s[:ml]
    sr=sr[:ml]
    return ml, 10*np.log10(np.sum(s**2)/(np.sum((s-sr)**2)+eps)+eps)

def Ada_test(model,teX,teY,tes):
    model.eval()
    ml=np.zeros(len(teX))
    SDRlist=np.zeros(len(teX))
    err=0.
    eps=1e-20
    bs = 50
    for i in range(0,len(teX),bs):        
        mbc=np.transpose(np.asarray(teX[i:i+bs]), (2,0,1))
        mb=Variable(torch.cuda.FloatTensor(np.abs(mbc)))
        teYh=model.forward(mb)
        mbY=Variable(torch.cuda.FloatTensor(np.transpose(np.array(teY[i:i+bs]), (2,0,1))))        
        err+=torch.sum(-mbY*torch.log(teYh+eps)-(1-mbY)*torch.log(1-teYh+eps))\
            .data.cpu().numpy()
        teSR=np.transpose(mbc*np.float32(teYh.data.cpu().numpy()), (2,0,1))
        print(teSR.shape)
        fake_func()
        #print(mbc.shape,teYh.shape,teSR.shape)
        for j in range(teSR.shape[2]):
            tesr=librosa.istft(teSR[:,:,j], hop_length=256)
            ml[i+j], SDRlist[i+j]=SDR(tes[i+j], tesr)
    return err, np.sum(ml*SDRlist/np.sum(ml))










