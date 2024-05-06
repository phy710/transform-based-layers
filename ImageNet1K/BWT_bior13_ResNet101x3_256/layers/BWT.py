#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 12:14:21 2022

@author: Zephyr
"""

import numpy as np
import torch
import pywt

from .common import find_min_power, SoftThresholding

def grayCode(n): 
    # Right Shift the number 
    # by 1 taking xor with  
    # original number 
    #return n ^ (n >> 1)   
    return [i^(i>>1) for i in range(n)]      

    
def permutation(A, axis = -1):
    if axis != -1:
        A = np.transpose(A, -1, axis)
    n = A.shape[-1]
    B = A[..., grayCode(n)]
    if axis != -1:
        B = np.transpose(B, -1, axis)
    return B

def bwt(x, wavelet='db1', axis=-1):
    (cA, cD) = pywt.dwt(x, wavelet, mode='periodization')
    if cA.shape[axis] <= 1:
        y = np.concatenate((cA, cD), axis=axis)
    else:
        y = np.concatenate((bwt(cA, wavelet, axis=axis), bwt(cD, wavelet, axis=axis)), axis=axis)
    return y

def get_bwt_matrix(n, wavelet='bior1.3'):
    return permutation(bwt(np.eye(n),  wavelet=wavelet, axis=-1))

def ptbwt(u, H = None, axis=-1):

    if axis != -1:
        u = torch.transpose(u, -1, axis)
    if H == None:
        n = u.shape[-1]
        H = torch.tensor(get_bwt_matrix(n), dtype=torch.float, device=u.device)
    y = u @ H
    if axis != -1:
        y = torch.transpose(y, -1, axis)
        
    return y

def ptibwt(u, H = None, axis=-1):
    if axis != -1:
        u = torch.transpose(u, -1, axis)
    if H == None:
        n = u.shape[-1]
        H = torch.tensor(np.linalg.inv(get_bwt_matrix(n)), dtype=torch.float, device=u.device)
    y = u @ H
    if axis != -1:
        y = torch.transpose(y, -1, axis)
        
    return y

class BWTConv2D(torch.nn.Module):
    def __init__(self, height, width, in_channels, out_channels, pods = 1, residual=True):
        super().__init__()
        self.height = height       
        self.width = width
        self.height_pad = find_min_power(self.height)  
        self.width_pad = find_min_power(self.width)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pods = pods
        self.conv = torch.nn.ModuleList([torch.nn.Conv2d(in_channels, out_channels, 1, bias=False) for i in range(self.pods)])
        self.ST = torch.nn.ModuleList([SoftThresholding((self.height_pad, self.width_pad)) for i in range(self.pods)])
        self.v = torch.nn.ParameterList([torch.rand((self.height_pad, self.width_pad)) for i in range(self.pods)])
        self.residual = residual
        self.register_buffer('H_height', torch.tensor(get_bwt_matrix(self.height_pad)*np.sqrt(self.height_pad), dtype=torch.float))
        self.register_buffer('H_width', torch.tensor(get_bwt_matrix(self.width_pad)*np.sqrt(self.width_pad), dtype=torch.float))#BWT transform matrix
        self.register_buffer('H_inv_height', torch.tensor(np.linalg.inv(get_bwt_matrix(self.height_pad)*np.sqrt(self.height_pad)), dtype=torch.float))
        self.register_buffer('H_inv_width', torch.tensor(np.linalg.inv(get_bwt_matrix(self.width_pad)*np.sqrt(self.width_pad)), dtype=torch.float))#BWT transform matrix

        
    def forward(self, x):
        height, width = x.shape[-2:]
        if height!= self.height or width!=self.width:
            raise Exception('({}, {})!=({}, {})'.format(height, width, self.height, self.width))
     
        f0 = x
        if self.width_pad>self.width or self.height_pad>self.height:
            f0 = torch.nn.functional.pad(f0, (0, self.width_pad-self.width, 0, self.height_pad-self.height))
        
        f1 = ptbwt(f0, self.H_width, axis=-1)
        f2 = ptbwt(f1, self.H_height, axis=-2)
        
        
        f3 = [self.v[i]*f2 for i in range(self.pods)]
        f4 = [self.conv[i](f3[i]) for i in range(self.pods)]
        f5 = [self.ST[i](f4[i]) for i in range(self.pods)]
        
        f6 = torch.stack(f5, dim=-1).sum(dim=-1)
        
        f7 = ptibwt(f6, self.H_inv_width, axis=-1)
        f8 = ptibwt(f7, self.H_inv_height, axis=-2)
        
        y = f8[..., :self.height, :self.width]
        
        if self.residual:
            y = y + x
        return y
