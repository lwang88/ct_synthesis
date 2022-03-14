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
from basicmodel.basic_x2.han import HAN as model1#pici,tongdao,chang,kuang,gao               
from basicmodel.basic_x2.cain  import CAIN as model2  #pici,tongdao,chang,kuang,gao         

class parallel_model(nn.Module):
    def __init__(self, args):
        super(parallel_model, self).__init__()
        self.model1 = model1(args)
        self.model2 = model2(args.depth)
        self.args =args
        
    def forward(self,x,m_items,train):#x:volume,(pici,chang,kuang,gao)
            yuan_volume=x.float().cuda(0)#1,2,3,4,5,6,7,8,9,10,11,12,13
            input_volume = yuan_volume[:,:,:,::2]#1,3,5,7,9,11,13
            lr_volume=input_volume[:,:,:,::2]#1,5,9,13
            b,c,k,g = lr_volume.shape
            interpolation_out = []
            interpolation_out_2_4 = []
            interpolation_out_05 = []
            for j in range(lr_volume.shape[3]-1):
                slice1=lr_volume[:,:,:,j]#1_frame
                slice1=slice1.unsqueeze(1)
                slice5=lr_volume[:,:,:,j+1]#5_frame
                slice5=slice5.unsqueeze(1)
                middle_slice3,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(slice1,slice5,m_items,train)#middle_slice is 3
                middle_slice2,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(slice1,middle_slice3,m_items,train)
                middle_slice4,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(middle_slice3,slice5,m_items,train)
                middle_slice_05,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(slice1,middle_slice2,m_items,train)
                middle_slice2 = middle_slice2.squeeze(1).unsqueeze(3)# remove channel and add height 
                middle_slice_input = middle_slice3.squeeze(1).unsqueeze(3)
                middle_slice4 = middle_slice4.squeeze(1).unsqueeze(3)
                middle_slice_05 = middle_slice_05.squeeze(1).unsqueeze(3)
                interpolation_out.append(middle_slice_input)#3,7,11
                interpolation_out_2_4.append(middle_slice2)#2,4,6,8...
                interpolation_out_2_4.append(middle_slice4)
                interpolation_out_05.append(middle_slice_05)
            interpolation_out=torch.cat(interpolation_out,dim=3)
            interpolation_out_2_4=torch.cat(interpolation_out_2_4,dim=3)
            interpolation_out_05=torch.cat(interpolation_out_05,dim=3)
            
            sr_lr_volume = lr_volume.unsqueeze(1).permute(0,2,1,3,4).view(-1,1,k,g)#1,5,9,13
            sr_out = self.model1(sr_lr_volume)#pici,chang,kuang,gao,1,3,5,7,9,11,13,15
            sr_out =sr_out[:,:,:,:-1]#8 to 7    1,3,5,7,9,11,13,
            b1,c1,k1,g1=input_volume.shape
            sr_out_retrain_left = sr_out.unsqueeze(1).permute(0,2,1,3,4).view(-1,1,k1,g1)
            sr_out_2_4_left = self.model1(sr_out_retrain_left)#pici,chang,kuang,gao ,1,2,3,4,5,6,7,8,9,10,11,12,13,14
            sr_out_2_4_left = sr_out_2_4_left[:,:,:,:-1]
            
            sr_out_retrain_right = sr_out.unsqueeze(1).permute(0,3,1,2,4).contiguous().view(-1,1,c1,g1)
            sr_out_2_4_right = self.model1(sr_out_retrain_right)#pici,chang,kuang,gao
            sr_out_2_4_right = sr_out_2_4_right.permute(0,2,1,3)#1,2,3,4,5,6,7,8,9,10,11,12,13,14
            sr_out_2_4_right = sr_out_2_4_right[:,:,:,:-1]#1,2,3,4,5,6,7,8,9,10,11,12,13,
            b2,c2,k2,g2 = sr_out_2_4_left.shape
            sr_out_05_left = sr_out_2_4_left.unsqueeze(1).permute(0,2,1,3,4).contiguous().view(-1,1,k2,g2) 
            sr_out_05_left = self.model1(sr_out_05_left)
            sr_out_05_left = sr_out_05_left[:,:,:,:-1]
            
            sr_out_05_right = sr_out_2_4_right.unsqueeze(1).permute(0,3,1,2,4).contiguous().view(-1,1,c2,g2)
            sr_out_05_right = self.model1(sr_out_05_right)#1,1.5,2,2.5,3,3.5.....13,13.5
            sr_out_05_right = sr_out_05_right.permute(0,2,1,3)
            sr_out_05_right = sr_out_05_right[:,:,:,:-1]#1,1.5,2,2.5,3,3.5.....13,
            
            sr_out_2_4_left=sr_out_2_4_left[:,:,:,1:-1:2]
            sr_out_2_4_right=sr_out_2_4_right[:,:,:,1:-1:2]
            N = interpolation_out_05.shape[-1]
            sr_out_05_left=sr_out_05_left[:,:,:,1:-1:2][:,:,:,:N]
            sr_out_05_right=sr_out_05_right[:,:,:,1:-1:2][:,:,:,:N]
            return  interpolation_out,sr_out,interpolation_out_2_4,sr_out_2_4_left,sr_out_2_4_right,m_items,interpolation_out_05,sr_out_05_left,sr_out_05_right, softmax_score_query,softmax_score_memory,separateness_loss,compactness_loss
            
