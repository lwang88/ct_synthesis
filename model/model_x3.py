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
from basicmodel.basic_x3.han import HAN as model1#pici,tongdao,chang,kuang,gao               
from basicmodel.basic_x3.cain  import CAIN as model2  #pici,tongdao,chang,kuang,gao       


class parallel_model(nn.Module):
    def __init__(self, args):
        super(parallel_model, self).__init__()
        self.model1 = model1(args)
        self.model2 = model2(args.depth)
        self.args =args
        
    def forward(self,x,m_items,train):#x:volume,(pici,chang,kuang,gao)
            yuan_volume=x.float().cuda(0)#1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19
            input_volume = yuan_volume[:,:,:,::3]#1,4,7,10,13,16,19
            lr_volume=input_volume[:,:,:,::3]#1,10,19
            b,c,k,g = lr_volume.shape
            interpolation_out = []
            interpolation_out_2_3_5_6_8_9 = []
            interpolation_out_03_06 = []
            for j in range(lr_volume.shape[3]-1):
                slice1=lr_volume[:,:,:,j]#1_frame
                slice1=slice1.unsqueeze(1)
                slice10=lr_volume[:,:,:,j+1]#10_frame
                slice10=slice10.unsqueeze(1)
                middle_slice4_7,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(slice1,slice10,m_items,train)#middle_slice is 3
                middle_slice2_3,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(slice1,middle_slice4_7[:,0,:,:].unsqueeze(1),m_items,train) #1+4->2,3
                middle_slice5_6,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(middle_slice4_7[:,0,:,:].unsqueeze(1),middle_slice4_7[:,1,:,:].unsqueeze(1),m_items,train) #4+7->5,6
                middle_slice8_9,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(middle_slice4_7[:,1,:,:].unsqueeze(1),slice10,m_items,train) #7+10->8,9
                middle_slice03_06,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(slice1,middle_slice2_3[:,0,:,:].unsqueeze(1),m_items,train) #1+2->03,07
                middle_slice4_7 = middle_slice4_7.permute(0,2,3,1)
                middle_slice2_3 = middle_slice2_3.permute(0,2,3,1)
                middle_slice5_6 = middle_slice5_6.permute(0,2,3,1)
                middle_slice8_9 = middle_slice8_9.permute(0,2,3,1)
                middle_slice03_06 = middle_slice03_06.permute(0,2,3,1) 
                interpolation_out.append(middle_slice4_7)
                interpolation_out_2_3_5_6_8_9.append(middle_slice2_3)
                interpolation_out_2_3_5_6_8_9.append(middle_slice5_6)
                interpolation_out_2_3_5_6_8_9.append(middle_slice8_9)
                interpolation_out_03_06.append(middle_slice03_06)
                
            interpolation_out=torch.cat(interpolation_out,dim=3)
            interpolation_out_2_3_5_6_8_9=torch.cat(interpolation_out_2_3_5_6_8_9,dim=3)
            interpolation_out_03_06=torch.cat(interpolation_out_03_06,dim=3)
            
            sr_out_left=[]
            sr_lr_volume = lr_volume.unsqueeze(1).permute(0,2,1,3,4).view(-1,1,k,g)#1,10,19
            sr_out = self.model1(sr_lr_volume)#1,4,7,10,13,16,19,21,23
            sr_out =sr_out[:,:,:,:-2]#1,4,7,10,13,16,19
            #for i in range(sr_out.shape[-1]):
            #    if (i%3!=0):
            #        sr_out_left.append(sr_out[:,:,:,i])
            #sr_out_left=torch.stack(sr_out_left,dim=-1)
            sr_out_left=sr_out
            sr_out_2_3_5_6_8_9_left=[]
            b1,c1,k1,g1=sr_out.shape##1,4,7,10,13
            sr_out_retrain_left = sr_out.unsqueeze(1).permute(0,2,1,3,4).view(-1,1,k1,g1)
            sr_out_retrain_left = self.model1(sr_out_retrain_left)#pici,chang,kuang,gao,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
            sr_out_retrain_left = sr_out_retrain_left[:,:,:,:-2]
            for i in range(sr_out.shape[-1]):
                if (i%3!=0):
                    sr_out_2_3_5_6_8_9_left.append(sr_out_retrain_left[:,:,:,i])
            sr_out_2_3_5_6_8_9_left=torch.stack(sr_out_2_3_5_6_8_9_left,dim=-1)
            
            sr_out_2_3_5_6_8_9_right=[]
            sr_out_retrain_right = sr_out.unsqueeze(1).permute(0,3,1,2,4).contiguous().view(-1,1,c1,g1)
            sr_out_retrain_right = self.model1(sr_out_retrain_right)#pici,chang,kuang,gao
            sr_out_retrain_right = sr_out_retrain_right.permute(0,2,1,3)
            sr_out_retrain_right = sr_out_retrain_right[:,:,:,:-2]
            for i in range(sr_out.shape[-1]):
                if (i%3!=0):
                    sr_out_2_3_5_6_8_9_right.append(sr_out_retrain_right[:,:,:,i])
            sr_out_2_3_5_6_8_9_right=torch.stack(sr_out_2_3_5_6_8_9_right,dim=-1)
            b2,c2,k2,g2 = sr_out_retrain_left.shape
            sr_out_03_06_left=[]
            sr_out_03_06_left_out = sr_out_retrain_left.unsqueeze(1).permute(0,2,1,3,4).view(-1,1,k2,g2)
            sr_out_03_06_left_out = self.model1(sr_out_03_06_left_out)
            sr_out_03_06_left_out = sr_out_03_06_left_out[:,:,:,:2]
            for i in range(sr_out_03_06_left_out.shape[-1]):
                if (i%3!=0):
                    sr_out_03_06_left.append(sr_out_03_06_left_out[:,:,:,i])
            sr_out_03_06_left=torch.stack(sr_out_03_06_left,dim=-1)
            
            sr_out_03_06_right=[]
            sr_out_05_right_out = sr_out_retrain_right.unsqueeze(1).permute(0,3,1,2,4).contiguous().view(-1,1,c2,g2)
            sr_out_retrain_right_out = self.model1(sr_out_05_right_out)
            sr_out_retrain_right_out = sr_out_retrain_right_out.permute(0,2,1,3)
            sr_out_retrain_right_out = sr_out_retrain_right_out[:,:,:,:2]
            for i in range(sr_out_retrain_right_out.shape[-1]):
                if (i%3!=0):
                    sr_out_03_06_right.append(sr_out_retrain_right_out[:,:,:,i])
            sr_out_03_06_right=torch.stack(sr_out_03_06_right,dim=-1)
            
            N=interpolation_out_03_06.shape[3]
            sr_out_03_06_left=sr_out_03_06_left[:,:,:,1:-1:2][:,:,:,:N]
            sr_out_03_06_right=sr_out_03_06_right[:,:,:,1:-1:2][:,:,:,:N]
            return  interpolation_out,sr_out_left,interpolation_out_2_3_5_6_8_9,sr_out_2_3_5_6_8_9_left,sr_out_2_3_5_6_8_9_right,m_items,interpolation_out_03_06,sr_out_03_06_left,sr_out_03_06_right, softmax_score_query,softmax_score_memory,separateness_loss,compactness_loss
            
