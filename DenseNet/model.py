# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:47:17 2021

@author: Dongil Ryu
"""
import torch
import torch.nn as nn

def conv_layer(is_BC, is_aug, in_f, k):
    if is_BC:
        layer = nn.Sequential(
            nn.BatchNorm2d(in_f),
            nn.ReLU(),
            nn.Conv2d(in_f, 4*k, kernel_size=1, padding=0),
            nn.BatchNorm2d(4*k),
            nn.ReLU(),
            nn.Conv2d(4*k, k, kernel_size=3, padding=1),
        )
    else:
        layer = nn.Sequential(
            nn.BatchNorm2d(in_f),
            nn.ReLU(),
            nn.Conv2d(in_f, k, kernel_size=3, padding=1),
        )
        
    if not is_aug:
        layer.add_module('dropout', nn.Dropout(p=0.2))
    return layer

def make_dense_block(is_BC, is_aug, k, layer_cnt, in_f):
    layer_list = nn.ModuleList()
    for i in range(layer_cnt):
        layer_list.append(conv_layer(is_BC, is_aug, in_f, k))
        in_f += k
    return layer_list, in_f

def pooling_layer(in_f, out_f, is_last=False):
    if not is_last:
        return nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=1, padding=0),
            nn.AvgPool2d(2, stride=2)
        )
    else:
        return nn.Sequential(
            nn.AvgPool2d(8, stride=8)
        )

class DenseNet(nn.Module):
    
    def __init__(self, L=40, k=12, theta=1, is_BC=False, is_aug=False, num_classes=100):
        super(DenseNet, self).__init__()
        self.L = L
        self.k = k
        self.theta = theta
        self.is_BC = is_BC
        self.is_aug = is_aug
        self.num_classes = num_classes
        
        in_f = 2 * k if is_BC else 16
        self.first_conv = nn.Conv2d(3, in_f, kernel_size=3, padding=1)
        
        self.dense_block_list = nn.ModuleList()
        self.pooling_layer_list = nn.ModuleList()
        for i in range(3):
            dense_block, in_f = make_dense_block(self.is_BC, self.is_aug, self.k, (self.L - 4) // 3, in_f)
            self.dense_block_list.append(dense_block)
            
            if i < 2:
                self.pooling_layer_list.append(pooling_layer(in_f, int(in_f * theta)))
                in_f = int(in_f * theta)
            else:
                self.pooling_layer_list.append(pooling_layer(-1, -1, True))
        
        self.decoder = nn.Linear(in_f, self.num_classes)

        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            # convolution kernel의 weight를 He initialization을 적용한다.
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                
                # bias는 상수 0으로 초기화 한다.
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)           
    
    def forward(self, inputs):
        out = self.first_conv(inputs)
        for dense_block, pooling_layer in zip(self.dense_block_list, self.pooling_layer_list):
            for layer in dense_block:
                tmp = layer(out)
                out = torch.cat((tmp, out), dim=1)
            out = pooling_layer(out)

        out = out.view(out.size(0), -1)
        out = self.decoder(out)
        return out
    