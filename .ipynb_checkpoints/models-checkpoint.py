#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:24:30 2019

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
#%matplotlib notebook
#bs=10

   
class plain_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        
#         self.conv1 = nn.Sequential(
#                    nn.Conv1d(1, 8, 5),
#                    nn.ReLU())
        
#         self.conv2 = nn.Sequential(
#                    nn.Conv1d(8, 16, 5, stride=2) ,
#                    nn.ReLU())     
        
#         self.gru1=nn.GRU(H*126, H2, 1) #gated recurrent unit (GRU) RNN  (input_size, hidden_size, num_layer) 
        self.conv1 = nn.Conv1d(1, 8, 5)
        self.conv2 = nn.Conv1d(8, 16, 5, stride=2)        
        
        self.fc1 = torch.nn.Linear(253*16, 1024) #253  # Applies a linear transformation to the incoming data
        self.fc2 = torch.nn.Linear(1024, 513)
        
        self.dropout = nn.Dropout(0.5) 

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)


    def forward(self, x):
        
        x = F.relu(self.conv1(x.view(-1, 1,513)))  
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        #  x.shape == (N, 16, 253)
        x = x.view(x.shape[0],-1)                 
        x = F.relu(self.fc1(x))    
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x


class gru(nn.Module):
    def __init__(self, H, H2, H3, D_out):
        super().__init__()
        
        self.conv1 = nn.Conv1d(2, 50, 5, stride=2)
        self.conv2 = nn.Conv1d(50, 50, 5, stride=2)        
        
        self.gru1=nn.GRU(50*126, 1024, 1) #gated recurrent unit (GRU) RNN  (input_size, hidden_size, num_layer) 
        
        self.fc1 = torch.nn.Linear(1024, 512)  # Applies a linear transformation to the incoming data
        self.fc2 = torch.nn.Linear(512, 513)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.fc1c.weight)
        nn.init.xavier_normal_(self.fc1i.weight)
        nn.init.xavier_normal_(self.fc2c.weight)
        nn.init.xavier_normal_(self.fc2i.weight)


    def forward(self, x):
        x = (F.relu(self.conv1(x)))        
        x = (F.relu(self.conv2(x)))
        x=x.view(-1, 10, 126*self.H)
        
        x, _ = self.gru1(x)
        x=x.view(-1, 10, 126*self.H)
                
        x = F.relu(self.fc1(x))       
        
        x = torch.sigmoid(self.fc2(x))


        return x
 

"""
.. Deep Residual Learning for Image Recognition:
    https://arxiv.org/abs/1512.03385
"""



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
