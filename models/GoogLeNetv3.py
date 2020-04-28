#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:56:01 2020

@author: lds
"""

import torch
import torch.nn as nn
from .Layers import Concat_Separable_Conv2d, Separable_Conv2d, Conv2d, Squeeze
from functools import partial

class GoogLeNetv3(nn.Module):
    def __init__(self, num_classes, mode='train'):
        super(GoogLeNetv3, self).__init__()
        self.num_classes = num_classes
        self.mode = mode     
        self.layers = nn.Sequential(
            Conv2d(3, 32, 3, stride=2),
            Conv2d(32, 32, 3, stride=1),
            Conv2d(32, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Conv2d(64, 80, kernel_size=3),
            Conv2d(80, 192, kernel_size=3, stride=2),
            Conv2d(192, 288, kernel_size=3, stride=1, padding=1),
            Inceptionv3(288, 64, 48, 64, 64, 96, 64, mode='1'), # 3a
            Inceptionv3(288, 64, 48, 64, 64, 96, 64, mode='1'), # 3b
            Inceptionv3(288, 0, 128, 384, 64, 96, 0, stride=2, pool_type='MAX', mode='1'), # 3c
            
            Inceptionv3(768, 192, 128, 192, 128, 192, 192, mode='2'), # 4a
            Inceptionv3(768, 192, 160, 192, 160, 192, 192, mode='2'), # 4b
            Inceptionv3(768, 192, 160, 192, 160, 192, 192, mode='2'), # 4c
            Inceptionv3(768, 192, 192, 192, 192, 192, 192, mode='2'), # 4d
            Inceptionv3(768, 0, 192, 320, 192, 192, 0, stride=2, pool_type='MAX', mode='2'), # 4e
            
            Inceptionv3(1280, 320, 384, 384, 448, 384, 192, mode='3'), # 5a
            Inceptionv3(2048, 320, 384, 384, 448, 384, 192, pool_type='MAX', mode='3'), # 5b
            nn.AvgPool2d(8, 1),
            Conv2d(2048, num_classes, kernel_size=1, output=True),
            Squeeze(),
        ) 
        if mode == 'train':
            self.aux = InceptionAux(768, num_classes)

    def forward(self, x):   
        for idx, layer in enumerate(self.layers):
            if(idx == 14 and self.mode == 'train'):
                aux = self.aux(x)
            x = layer(x)
        if self.mode == 'train':
            return x, aux
        else:
            return x
    
    def init_weights(self, init_mode='VGG'):
        def init_function(m, init_mode):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_mode == 'VGG':
                    torch.nn.init.normal_(m.weight, mean=0, std=0.01)
                elif init_mode == 'XAVIER': 
                    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                    std = (2.0 / float(fan_in + fan_out)) ** 0.5
                    a = (3.0)**0.5 * std
                    with torch.no_grad():
                        m.weight.uniform_(-a, a)
                elif init_mode == 'KAMING':
                     torch.nn.init.kaiming_uniform_(m.weight)
                
                m.bias.data.fill_(0)    
        _ = self.apply(partial(init_function, init_mode=init_mode))
    
class Inceptionv3(nn.Module):
    def __init__(self, input_channel, conv1_channel, conv3_reduce_channel,
                 conv3_channel, conv3_double_reduce_channel, conv3_double_channel, pool_reduce_channel, stride=1, pool_type='AVG', mode='1'):
        '''
        pool_type : TYPE, ['AVG', 'MAX']
            DESCRIPTION. The default is 'AVG'.

        '''
        super(Inceptionv3, self).__init__()
        self.stride = stride
        if stride == 2:
            padding_conv3 = 0
            padding_conv7 = 2
        else:
            padding_conv3 = 1
            padding_conv7 = 3
        if conv1_channel != 0:
            self.conv1 = Conv2d(input_channel, conv1_channel, kernel_size=1)
        else:
            self.conv1 = None
        self.conv3_reduce = Conv2d(input_channel, conv3_reduce_channel, kernel_size=1)
        if mode == '1':
            self.conv3 = Conv2d(conv3_reduce_channel, conv3_channel, kernel_size=3, stride=stride, padding=padding_conv3)
            self.conv3_double1 = Conv2d(conv3_double_reduce_channel, conv3_double_channel, kernel_size=3, padding=1)
            self.conv3_double2 = Conv2d(conv3_double_channel, conv3_double_channel, kernel_size=3, stride=stride, padding=padding_conv3)
        elif mode == '2':
            self.conv3 = Separable_Conv2d(conv3_reduce_channel, conv3_channel, kernel_size=7, stride=stride, padding=padding_conv7)
            self.conv3_double1 = Separable_Conv2d(conv3_double_reduce_channel, conv3_double_channel, kernel_size=7, padding=3)
            self.conv3_double2 = Separable_Conv2d(conv3_double_channel, conv3_double_channel, kernel_size=7, stride=stride, padding=padding_conv7)
        elif mode == '3':
            self.conv3 = Concat_Separable_Conv2d(conv3_reduce_channel, conv3_channel, kernel_size=3, stride=stride, padding=1)
            self.conv3_double1 = Conv2d(conv3_double_reduce_channel, conv3_double_channel, kernel_size=3, padding=1)
            self.conv3_double2 = Concat_Separable_Conv2d(conv3_double_channel, conv3_double_channel, kernel_size=3, stride=stride, padding=1)
        
        self.conv3_double_reduce = Conv2d(input_channel, conv3_double_reduce_channel, kernel_size=1)
        if pool_type == 'MAX':
            self.pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=padding_conv3)
        elif pool_type == 'AVG':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=padding_conv3)
        if pool_reduce_channel != 0:
            self.pool_reduce = Conv2d(input_channel, pool_reduce_channel, kernel_size=1)
        else:
            self.pool_reduce = None
    
    def forward(self, x):

        output_conv3 = self.conv3(self.conv3_reduce(x))
        output_conv3_double = self.conv3_double2(self.conv3_double1(self.conv3_double_reduce(x)))
        if self.pool_reduce != None:
            output_pool = self.pool_reduce(self.pool(x))
        else:
            output_pool = self.pool(x)
            
        if self.conv1 != None:
            output_conv1 = self.conv1(x)        
            outputs = torch.cat([output_conv1, output_conv3, output_conv3_double, output_pool], dim=1)
        else:
            outputs = torch.cat([output_conv3, output_conv3_double, output_pool], dim=1)
        return outputs  

class InceptionAux(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(InceptionAux, self).__init__()
        self.layers = nn.Sequential(
            nn.AvgPool2d(5, 3),
            Conv2d(input_channel, 128, 1),
            Conv2d(128, 1024, kernel_size=5),
            Conv2d(1024, num_classes, kernel_size=1, output=True),
            Squeeze()
            )
    
    def forward(self, x):
        x = self.layers(x)
        return x

if __name__ == '__main__':
    net = GoogLeNetv3(1000).cuda()
    from torchsummary import summary
    summary(net, (3, 299, 299))
