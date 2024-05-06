#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 12:14:21 2022

@author: Zephyr
"""

import numpy as np
import torch
from scipy.fft import dct, idct

from .common import SoftThresholding

def discrete_cosine_transform(u, axis=-1):
    if axis != -1:
        u = torch.transpose(u, -1, axis)
    
    n = u.shape[-1]
    D = torch.tensor(dct(np.eye(n), axis=-1), dtype=torch.float, device=u.device)
    y = u@D
    
    if axis != -1:
        y = torch.transpose(y, -1, axis)
        
    return y

def inverse_discrete_cosine_transform(u, axis=-1):
    if axis != -1:
        u = torch.transpose(u, -1, axis)
    
    n = u.shape[-1]
    D = torch.tensor(idct(np.eye(n), axis=-1), dtype=torch.float, device=u.device)
    y = u@D
    
    if axis != -1:
        y = torch.transpose(y, -1, axis)
        
    return y
    
class DCT1D(torch.nn.Module):
    """
    1D DCT layer. We apply analysis along the last axis. 
    num_features: Length of the last axis.
    residual: Apply shortcut connection or not
    retain_DC: Retain DC channel (the first channel) or not
    """
    def __init__(self, num_features, residual=True, retain_DC=False):
        super().__init__()
        self.num_features = num_features
        if retain_DC:
            self.ST = SoftThresholding(self.num_features-1)    
            self.v = torch.nn.Parameter(torch.rand(self.num_features-1))
        else:
            self.ST = SoftThresholding(self.num_features)    
            self.v = torch.nn.Parameter(torch.rand(self.num_features))
        self.residual = residual
        self.retain_DC = retain_DC
        #with torch.no_grad():
        #    self.dense.weight.copy_(torch.eye(num_features))
         
    def forward(self, x):
        num_features = x.shape[-1]
        if num_features!= self.num_features:
            raise Exception('{}!={}'.format(num_features, self.num_features))
        f0 = x
        f1 = discrete_cosine_transform(f0)
        if self.retain_DC:
            f2 = self.v*f1[..., 1:]
            f3 = self.ST(f2)
            f4 = inverse_discrete_cosine_transform(torch.cat((f1[..., 0:1], f3), axis=-1))
        else:
            f2 = self.v*f1
            f3 = self.ST(f2)
            f4 = inverse_discrete_cosine_transform(f3)
        y = f4
        if self.residual:
            y = y + x
        return y

class DCTConv2D(torch.nn.Module):
    def __init__(self, height, width, in_channels, out_channels, pods = 1, residual=True):
        super().__init__()
        self.height = height       
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pods = pods
        self.conv = torch.nn.ModuleList([torch.nn.Conv2d(in_channels, out_channels, 1, bias=False) for i in range(self.pods)])
        self.ST = torch.nn.ModuleList([SoftThresholding((self.height, self.width)) for i in range(self.pods)])
        self.v = torch.nn.ParameterList([torch.rand((self.height, self.width)) for i in range(self.pods)])
        self.residual = residual
        
    def forward(self, x):
        height, width = x.shape[-2:]
        if height!= self.height or width!=self.width:
            raise Exception('({}, {})!=({}, {})'.format(height, width, self.height, self.width))
     
        f0 = x
           
        f1 = discrete_cosine_transform(f0, axis=-1)
        f2 = discrete_cosine_transform(f1, axis=-2)

        f3 = [self.v[i]*f2 for i in range(self.pods)]
        f4 = [self.conv[i](f3[i]) for i in range(self.pods)]
        f5 = [self.ST[i](f4[i]) for i in range(self.pods)]
        
        f6 = torch.stack(f5, dim=-1).sum(dim=-1)
        
        f7 = inverse_discrete_cosine_transform(f6, axis=-1)
        f8 = inverse_discrete_cosine_transform(f7, axis=-2)
        
        y = f8
        
        if self.residual:
            y = y + x
        return y