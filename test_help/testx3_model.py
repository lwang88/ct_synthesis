# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:06:03 2020
ct_interpolation_super_resolution
@author: wangliang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np         
from basicmodel.basic_x3.han  import HAN as model1           
from basicmodel.basic_x3.cain  import CAIN as model2   

class parallel_model(nn.Module):
    def __init__(self, args):
        super(parallel_model, self).__init__()
        self.model1 = model1(args)
        self.model2 = model2()
        self.args=args
        
    def forward(self,x,m_items,train=False):#x:volume,(pici,chang,kuang,gao)
            lr_volume=x.float().cuda(0)#1,4,7,11,14
            b,c,k,g = lr_volume.shape
            interpolate_out=[]
            for j in range(lr_volume.shape[3]-1):
                slice0=lr_volume[:,:,:,j]
                slice0=slice0.unsqueeze(1)
                slice1=lr_volume[:,:,:,j+1]
                slice1=slice1.unsqueeze(1)
                middle_slice_interpolate,  _, _, _, _, _, _,_,_,_= self.model2(slice0,slice1,m_items,train)
                for h in range(middle_slice_interpolate.shape[1]):
                    interpolate_out.append(middle_slice_interpolate[:,h,:,:])
            middle_slice_sr_left=self.model1(lr_volume.unsqueeze(1).permute(0,2,1,3,4).view(-1,1,k,g))[:,:,:,:-2]
            b1,_,_,g1=middle_slice_sr_left.shape
            middle_slice_sr_left=middle_slice_sr_left.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,k,g1).permute(0,2,1,3,4).squeeze(1)
            middle_slice_sr_right=self.model1(lr_volume.unsqueeze(1).permute(0,3,1,2,4).view(-1,1,c,g))[:,:,:,:-2]
            b1,_,_,g1=middle_slice_sr_right.shape
            middle_slice_sr_right=middle_slice_sr_right.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,k,g1).permute(0,2,1,3,4).squeeze(1).permute(0,2,1,3)
            res_volume=[]
            j=0
            for i in range(min(middle_slice_sr_left.shape[3],middle_slice_sr_right.shape[3])):
                if i%self.args.upscale!=0:
                    res_volume.append((interpolate_out[j]+middle_slice_sr_left[:,:,:,i]+middle_slice_sr_right[:,:,:,i])/3)
                    j+=1
                else:
                    res_volume.append((middle_slice_sr_left[:,:,:,i]+middle_slice_sr_right[:,:,:,i])/2)
            res_volume=torch.stack(res_volume,3)
            return middle_slice_sr_right