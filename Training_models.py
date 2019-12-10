#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:21:11 2019

@author: haici
"""
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

from nn_models import AudioCoLearning, Audio_NonCoLearning  
from Testing_models import AudioCoLearning_Uni_Test, AudioCoLearning_Bi_Test, NonCoLearning_Bi_Test, SDR


# Bi-model
def train_Siam():
    bs = 10
    maxSDR = 0
    lambda1 = 1.
    epoch_siam = []
    for epoch in range(maxEpoch):    
        model.train()
        
        k1=np.random.permutation(10) # k1=np.random.permutation(bs)
        k2=np.roll(k1,1) #random job
        begin_ep = time.time()
        for i in range(0,len(trX),bs*100):
            
            mb1=Variable(torch.cuda.FloatTensor((np.abs(np.transpose(np.asarray([trX[kk] for kk in i+k1]), (2,0,1))))),
                        requires_grad=False)
            mb2=Variable(torch.cuda.FloatTensor((np.abs(np.transpose(np.asarray([trX[kk] for kk in i+k2]), (2,0,1))))),
                        requires_grad=False)
            mbY1=Variable(torch.cuda.FloatTensor((np.transpose(np.array([trY[kk] for kk in i+k1]), (2,0,1)))), requires_grad=False)
            mbY2=Variable(torch.cuda.FloatTensor((np.transpose(np.array([trY[kk] for kk in i+k2]), (2,0,1)))), requires_grad=False)
            trYh1, trYh2, clf, if1, if2, h112=model(mb1, mb2)
            print(h112.shape)
    
            err=torch.sum(-mbY1*torch.log(trYh1+eps)-(1-mbY1)*torch.log(1-trYh1+eps))+\
                    torch.sum(-mbY2*torch.log(trYh2+eps)-(1-mbY2)*torch.log(1-trYh2+eps))
            #\
    #                 +lambda1*(torch.sum(clf)/torch.sum(if1)-1.5)**2+lambda1*(torch.sum(clf)/torch.sum(if2)-1.5)**2
            errt[epoch]+=err.data.cpu().numpy()
            optimizer.zero_grad()
            err.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),.0001)
            optimizer.step()
        print('ep: {}\t train error: {}'.format(epoch, errt[epoch]))
        _, SDR1, SDR2=AudioCoLearning_Bi_Test(model) # test
        print(SDR1, SDR2)
        end_ep = time.time()
        print(end_ep - begin_ep)
        epoch_siam.append((SDR1+SDR2)/2)
        np.save('epoch_siam.npy', epoch_siam)
        if (SDR1+SDR2)/2 >maxSDR:
            maxSDR=(SDR1+SDR2)/2
            
            #torch.save(model.state_dict(), 'SS_CNN_GRU_Siamese_Bi.model')  

def train_NonSiam_Uni():
    bs=10
    maxSDR=0
    lambda1=1.
    epoch_uni = []
    for epoch in range(maxEpoch):    
        model.train()
        
        k1=np.random.permutation(10) # k1=np.random.permutation(bs)
        k2=np.roll(k1,1) #random job
        for i in range(0,len(trX),bs*100):
            mb=Variable(torch.cuda.FloatTensor((np.abs(np.transpose(np.asarray([trX[kk] for kk in i+k1]), (2,0,1))))),
                        requires_grad=False)
            mbc=np.transpose(np.asarray(trX[i:i+bs]), (2,0,1))
            mbY=Variable(torch.cuda.FloatTensor((np.transpose(np.array([trY[kk] for kk in i+k1]), (2,0,1)))), requires_grad=False)
            
            trYh=model.forward_uni(mb)      
            err=torch.sum(-mbY*torch.log(trYh+eps)-(1-mbY)*torch.log(1-trYh+eps))
            
    #         err=torch.sum(-mbY1*torch.log(trYh1+eps)-(1-mbY1)*torch.log(1-trYh1+eps))+\
    #                 torch.sum(-mbY2*torch.log(trYh2+eps)-(1-mbY2)*torch.log(1-trYh2+eps))
    
            errt[epoch]+=err.data.cpu().numpy()
            optimizer.zero_grad()
            err.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),.0001)
            optimizer.step()
    
        print('ep: {}\t train error: {}'.format(epoch, errt[epoch]))
        _, SDR_uni=AudioCoLearning_Uni_Test(model) # test
        print(SDR_uni)
        epoch_uni.append(SDR_uni)
        np.save('epoch_non_uni.npy', epoch_uni)
        if SDR_uni >maxSDR:
            maxSDR=SDR_uni
            torch.save(model.state_dict(), 'SS_CNN_GRU_Siamese_Bi.model')  
            
            
def Non_Siam_Bi(trX, trY, teX, teY, tes):
    lookback=30
    Hcnn=50
    Hrnn=1024
    Hcl=512
    # nL=2
    #bs=10
    eps=1e-20
    maxEpoch = 50
    errt=np.zeros(maxEpoch, dtype=np.float32)
    maxSDR=0
    lambda1=1.
    epoch_Non = []
    
    model_ns = Audio_NonCoLearning (Hcnn, Hrnn, Hcl, 513, bs=2).cuda()
    optimizer= torch.optim.Adam(model_ns.parameters(), lr=0.001)#, betas=[0.9, 0.999])
    
#     k1=np.random.permutation(10) # k1=np.random.permutation(bs)
#     k2=np.roll(k1,1) #random job
#     trX_f1=[] # flattened trX 
#     trX_f2=[] # flattened trX in another order
    
#     for i in range(0,len(trX),10):
#         trx1_bs= np.asarray([trX[kk] for kk in i+k1])
#         fx1=trx1_bs.transpose(1,0,2).reshape(len(trX[i]), 10*len(trX[i][0])) # 3d to 2d (513,:)
#         trX_f1.append(fx1)    
        
#         trx2_bs = np.asarray([trX[kk] for kk in i+k2])
#         fx2=trx2_bs.transpose(1,0,2).reshape(len(trX[i]), 10*len(trX[i][0]))
#         trX_f2.append(fx2)
        
    
    for epoch in range(maxEpoch):    
        model_ns.train()
        k1=np.random.permutation(10) # k1=np.random.permutation(bs)
        k2=np.roll(k1,1) #random job
        
        for i in range(0, int(len(trX)/100), 10): 
            #stack two sequences of trX together
            trx1_bs= np.asarray([trX[kk] for kk in i+k1])
            fx1=trx1_bs.transpose(1,0,2).reshape(len(trX[i]), 10*len(trX[i][0])) # 3d to 2d (513,:)
            trx2_bs = np.asarray([trX[kk] for kk in i+k2])
            fx2=trx2_bs.transpose(1,0,2).reshape(len(trX[i]), 10*len(trX[i][0]))
            stacked_x = np.stack((fx1,fx2))
            
            mbc=np.transpose(np.asarray(stacked_x), (2,0,1))
            mb=Variable(torch.cuda.FloatTensor(np.abs(np.transpose(stacked_x, (2,0,1)))),
                        requires_grad=False)
            
            #stacked_y = np.stack((trY_f1[i],trY_f2[i]))
#             stacked_y = trY_stacked[i]
#             mbY=Variable(torch.cuda.FloatTensor(np.abs(np.transpose(stacked_y, (2,0,1)))),
#                         requires_grad=False)
            mbY1 = Variable(torch.cuda.FloatTensor(np.transpose(np.abs(np.asarray([trY[kk] for kk in i+k1])), (2,0,1))), requires_grad=False)
            mbY2 = Variable(torch.cuda.FloatTensor(np.transpose(np.abs(np.asarray([trY[kk] for kk in i+k2])), (2,0,1))), requires_grad=False)
                    

            trYh = model_ns.forward(mb)
            #print('h112',h112.shape)
            #print('h12',h12.shape)

#             err=torch.sum(-mbY*torch.log(trYh+eps)-(1-mbY)*torch.log(1-trYh+eps))
            
            err=torch.sum(-mbY1*torch.log(trYh+eps)-(1-mbY1)*torch.log(1-trYh+eps))+\
                    torch.sum(-mbY2*torch.log(trYh+eps)-(1-mbY2)*torch.log(1-trYh+eps))
   
            errt[epoch]+=err.data.cpu().numpy()
            optimizer.zero_grad()
            err.backward()
            torch.nn.utils.clip_grad_norm_(model_ns.parameters(),.0001)
            optimizer.step()


        print('ep: {}\t train error: {}'.format(epoch, errt[epoch]))
        _, SDR_non=NonCoLearning_Bi_Test(model_ns,teX, teY, tes) # test
        print(SDR_non)
        epoch_Non.append(SDR_non)
        np.save("epoch_Non_10.npy", epoch_Non)
        if SDR_non >maxSDR:
            maxSDR=SDR_non
            torch.save(model_ns.state_dict(), 'SS_CNN_GRU_NonSiamese_Bi.model')  