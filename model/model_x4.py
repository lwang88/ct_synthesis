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
from basicmodel.basic_x4.han import HAN as model1#pici,tongdao,chang,kuang,gao               
from basicmodel.basic_x4.cain  import CAIN as model2  #pici,tongdao,chang,kuang,gao         


class parallel_model(nn.Module):
    def __init__(self, args):
        super(parallel_model, self).__init__()
        self.model1 = model1(args)
        self.model2 = model2(args.depth)
        self.args =args
        
    def forward(self,x,m_items,train):#x:volume,(pici,chang,kuang,gao)
            yuan_volume=x.float().cuda(0)#1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,
            input_volume = yuan_volume[:,:,:,::4]#1,5,9,13,17,21,25,29,33
            lr_volume=input_volume[:,:,:,::4]#1,17,33
            b,c,k,g = lr_volume.shape
            interpolation_out = []
            interpolation_out_2_3_4_6_7_8_10_11_12_14_15_16 = []
            interpolation_out_02_05_08 = []
            for j in range(lr_volume.shape[3]-1):
                slice1=lr_volume[:,:,:,j]#1_frame
                slice1=slice1.unsqueeze(1)
                slice17=lr_volume[:,:,:,j+1]#9_frame
                slice17=slice5.unsqueeze(1)
                middle_slice5_9_13,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(slice1,slice17,m_items,train)
                middle_slice2_3_4,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(slice1,middle_slice5_9_13[:,0,:,:],m_items,train) # 1+5->2,3,4
                middle_slice6_7_8,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(middle_slice5_9_13[:,0,:,:],middle_slice5_9_13[:,1,:,:],m_items,train) # 5+9->6,7,8
                middle_slice10_11_12,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(middle_slice5_9_13[:,1,:,:],middle_slice5_9_13[:,2,:,:],m_items,train)#9+13->10,11,12
                middle_slice14_15_16,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(middle_slice5_9_13[:,2,:,:],slice17,m_items,train) # 13+17->14,15,16
                middle_slice02_05_08,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(slice1,middle_slice2_3_4[:,0,:,:],m_items,train)
                
                middle_slice5_9_13 = middle_slice4_5_6.permute(0,2,3,1)
                middle_slice2_3_4 = middle_slice2_3_4.permute(0,2,3,1)
                middle_slice6_7_8 = middle_slice6_7_8.permute(0,2,3,1)
                middle_slice10_11_12 = middle_slice10_11_12.permute(0,2,3,1)
                middle_slice14_15_16 = middle_slice14_15_16.permute(0,2,3,1)
                middle_slice02_05_08 = middle_slice02_05_08.permute(0,2,3,1)
                
                interpolation_out.append(middle_slice5_9_13)
                interpolation_out_2_3_4_6_7_8_10_11_12_14_15_16.append(middle_slice2_3_4)
                interpolation_out_2_3_4_6_7_8_10_11_12_14_15_16.append(middle_slice6_7_8)
                interpolation_out_2_3_4_6_7_8_10_11_12_14_15_16.append(middle_slice10_11_12)
                interpolation_out_2_3_4_6_7_8_10_11_12_14_15_16.append(middle_slice14_15_16)
                interpolation_out_02_05_08.append(middle_slice02_05_08)

            sr_out_left=[]
            sr_lr_volume = lr_volume.unsqueeze(1).permute(0,2,1,3,4).view(-1,1,k,g)#1,17,33
            sr_out = self.model1(sr_lr_volume)#pici,chang,kuang,gao,1,3,5,7,9,11,13,15
            sr_out =sr_out[:,:,:,:-3]#1,3,5,7,9,
            for i in range(sr_out.shape[-1]):
                if (i%4!=0):
                    sr_out_left.append(sr_out[:,:,:,i])
            sr_out_left=torch.stack(sr_out_left,dim=-1)
            
            sr_out_2_3_4_6_7_8_10_11_12_14_15_16_left
            b1,c1,k1,g1=sr_out.shape
            sr_out_retrain_left = sr_out.unsqueeze(1).permute(0,2,1,3,4).view(-1,1,k1,g1)
            sr_out_retrain_left = self.model1(sr_out_retrain_left)#pici,chang,kuang,gao
            sr_out_retrain_left = sr_out_retrain_left[:,:,:,:-3]
            for i in range(sr_out.shape[-1]):
                if (i%4!=0):
                    sr_out_2_3_4_6_7_8_10_11_12_14_15_16_left.append(sr_out_retrain_left[:,:,:,i])
            sr_out_2_3_4_6_7_8_10_11_12_14_15_16_left=torch.stack(sr_out_2_3_4_6_7_8_10_11_12_14_15_16_left,dim=-1)
            
            sr_out_2_3_4_6_7_8_10_11_12_14_15_16_right=[]
            sr_out_retrain_right = sr_out.unsqueeze(1).permute(0,3,1,2,4).contiguous().view(-1,1,c1,g1)
            sr_out_retrain_right = self.model1(sr_out_retrain_right)#pici,chang,kuang,gao
            sr_out_retrain_right = sr_out_retrain_right.permute(0,2,1,3)
            sr_out_retrain_right = sr_out_retrain_right[:,:,:,:-3]
            for i in range(sr_out_retrain_right.shape[-1]):
                if (i%4!=0):
                    sr_out_2_3_4_6_7_8_10_11_12_14_15_16_right.append(sr_out_retrain_right[:,:,:,i])
            sr_out_2_3_4_6_7_8_10_11_12_14_15_16_right=torch.cat(sr_out_2_3_4_6_7_8_10_11_12_14_15_16_right,dim=-1)
            
            sr_out_02_05_08_left=[]
            sr_out_05_left = sr_out_retrain_left.unsqueeze(1).permute(0,2,1,3,4).view(-1,1,k1,4)
            sr_out_05_left = self.model1(sr_out_05_left)
            sr_out_05_left = sr_out_05_left[:,:,:,:-3]
            for i in range(sr_out_05_left.shape[-1]):
                if (i%4!=0):
                    sr_out_02_05_08_left.append(sr_out_05_left[:,:,:,i])
            sr_out_02_05_08_left=torch.cat(sr_out_02_05_08_left,dim=-1)

            sr_out_02_05_08_right=[]
            sr_out_05_right = sr_out_retrain_right.unsqueeze(1).permute(0,3,1,2,4).contiguous().view(-1,1,c1,4)
            sr_out_05_right = self.model1(sr_out_05_right)
            sr_out_05_right = sr_out_05_right.permute(0,2,1,3)
            sr_out_05_right = sr_out_05_right[:,:,:,:-3]
            for i in range(sr_out_05_right.shape[-1]):
                if (i%4!=0):
                    sr_out_02_05_08_right.append(sr_out_05_right[:,:,:,i])
            sr_out_02_05_08_right=torch.stack(sr_out_02_05_08_right,dim=-1)
            
            N=interpolation_out_02_05_08.shape[3]
            sr_out_05_left=sr_out_05_left[0:N,:,:,1:-1:2]
            sr_out_05_right=sr_out_05_right[0:N,:,:,1:-1:2]
            return  interpolation_out,sr_out_left,interpolation_out_2_3_4_6_7_8_10_11_12_14_15_16,sr_out_2_3_4_6_7_8_10_11_12_14_15_16_left,sr_out_2_3_4_6_7_8_10_11_12_14_15_16_right,m_items,interpolation_out_02_05_08,sr_out_02_05_08_left,sr_out_02_05_08_right, softmax_score_query,softmax_score_memory,separateness_loss,compactness_loss
            
