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
            yuan_volume=x.float().cuda(0)#1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,.¡£¡£33
            input_volume = yuan_volume[:,:,:,::4]#1,5,9,13,17,21,25,29,33
            lr_volume=input_volume[:,:,:,::4]#1,17,33
            b,c,k,g = lr_volume.shape
            interpolation_out = []
            interpolation_out_2_3_4_6_7_8_10_11_12_14_15_16 = []
            interpolation_out_02_05_08 = []
            slice1_17=lr_volume[:,:,:,:-1].permute(0,3,1,2).reshape(-1,1,c,k)#1_frame,17_frame
            slice17_33=lr_volume[:,:,:,1:].permute(0,3,1,2).reshape(-1,1,c,k)#17_frame,33_frame
            slice5_9_13_21_25_29,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(slice1_17,slice17_33,m_items,train)
            slice5_9_13_21_25_29 = slice5_9_13_21_25_29.contiguous().view(-1,1,c,k)
            slice1_5_9_13_17_21_25_29=input_volume[:,:,:,:-1].permute(0,3,1,2).reshape(-1,1,c,k)
            slice5_9_13_17_21_25_29_33=input_volume[:,:,:,1:].permute(0,3,1,2).reshape(-1,1,c,k)
            
            slice2_3_4_6_7_8_10_11_12,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(slice1_5_9_13_17_21_25_29,slice5_9_13_17_21_25_29_33,m_items,train)
            slice2_3_4_6_7_8_10_11_12 = slice2_3_4_6_7_8_10_11_12.contiguous().view(-1,1,c,k)
            slice1_2_3_4_5_6_7=[]
            slice2_3_4_5_6_7_8=[]
            for i in range(slice1_5_9_13_17_21_25_29.shape[0]):
                slice1_2_3_4_5_6_7.append(slice1_5_9_13_17_21_25_29[i,:,:,:])
                slice1_2_3_4_5_6_7.append(slice2_3_4_6_7_8_10_11_12[3*i,:,:,:])
                slice1_2_3_4_5_6_7.append(slice2_3_4_6_7_8_10_11_12[3*i+1,:,:,:])
                slice1_2_3_4_5_6_7.append(slice2_3_4_6_7_8_10_11_12[3*i+2,:,:,:])
            for i in range(slice5_9_13_17_21_25_29_33.shape[0]):
                slice2_3_4_5_6_7_8.append(slice2_3_4_6_7_8_10_11_12[3*i,:,:,:])
                slice2_3_4_5_6_7_8.append(slice2_3_4_6_7_8_10_11_12[3*i+1,:,:,:])
                slice2_3_4_5_6_7_8.append(slice2_3_4_6_7_8_10_11_12[3*i+2,:,:,:])
                slice2_3_4_5_6_7_8.append(slice5_9_13_17_21_25_29_33[i,:,:,:])
            slice1_2_3_4_5_6_7 = torch.cat(slice1_2_3_4_5_6_7,0).unsqueeze(1)
            slice2_3_4_5_6_7_8 = torch.cat(slice2_3_4_5_6_7_8,0).unsqueeze(1)
            slice02_05_08,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(slice1_2_3_4_5_6_7,slice2_3_4_5_6_7_8,m_items,train) #1+2->03,07
            slice02_05_08 = slice02_05_08.contiguous().view(-1,1,c,k)
            
            interpolation_out=slice5_9_13_21_25_29.reshape(b,-1,c,k).permute(0,2,3,1)
            interpolation_out_2_3_4_6_7_8=slice2_3_4_6_7_8_10_11_12.reshape(b,-1,c,k).permute(0,2,3,1)
            interpolation_out_02_05_08=slice02_05_08.reshape(b,-1,c,k).permute(0,2,3,1)

            sr_lr_volume = lr_volume.unsqueeze(1).permute(0,2,1,3,4).reshape(-1,1,k,g)#1,17,33
            sr_out = self.model1(sr_lr_volume)#1,5,9,13,17,21,25,29,33,37,41,45
            sr_out =sr_out[:,:,:,:-3]#1,5,9,13,17,21,25,29,33
            b1,_,_,g1=sr_out.shape
            sr_left=sr_out.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,k,-1).permute(0,2,1,3,4).squeeze(1)
            
            sr_lr_volume = lr_volume.unsqueeze(1).permute(0,3,1,2,4).reshape(-1,1,c,g)#1,17,33
            sr_out = self.model1(sr_lr_volume)#1,5,9,13,17,21,25,29,33,37,41,45
            sr_out =sr_out[:,:,:,:-3]#1,5,9,13,17,21,25,29,33
            b1,_,_,g1=sr_out.shape
            sr_right=sr_out.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,c,-1).permute(0,2,1,3,4).squeeze(1).permute(0,2,1,3)
            
            sr_out_left=input_volume
            sr_out_retrain_left = sr_out_left.unsqueeze(1).permute(0,2,1,3,4).contiguous().view(-1,1,k,g1)
            sr_out_retrain_right = sr_out_left.unsqueeze(1).permute(0,3,1,2,4).contiguous().view(-1,1,c,g1)
            sr_out_retrain_left_right = torch.cat((sr_out_retrain_left,sr_out_retrain_right),0)
            
            sr_out_2_3_4_6_7_8_left_right = self.model1(sr_out_retrain_left_right)[:,:,:,:-3]#pici,chang,kuang,gao,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21
            sr_out_02_05_08_left_right = self.model1(sr_out_2_3_4_6_7_8_left_right)[:,:,:,:-3]
            
            sr_out_2_3_4_6_7_8_left_slice,sr_out_2_3_4_6_7_8_right_slice = sr_out_2_3_4_6_7_8_left_right.chunk(2,dim=0)
            sr_out_02_05_08_left_slice,sr_out_02_05_08_right_slice = sr_out_02_05_08_left_right.chunk(2,dim=0)
            sr_out_2_3_4_6_7_8_left_slice=sr_out_2_3_4_6_7_8_left_slice.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,c,-1).permute(0,2,1,3,4).squeeze(1)
            sr_out_2_3_4_6_7_8_right_slice=sr_out_2_3_4_6_7_8_right_slice.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,k,-1).permute(0,2,1,3,4).squeeze(1)
            sr_out_02_05_08_left_slice=sr_out_02_05_08_left_slice.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,c,-1).permute(0,2,1,3,4).squeeze(1)
            sr_out_02_05_08_right_slice=sr_out_02_05_08_right_slice.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,k,-1).permute(0,2,1,3,4).squeeze(1)
            sr_out_2_3_4_6_7_8_right_slice=sr_out_2_3_4_6_7_8_right_slice.permute(0,2,1,3)
            sr_out_02_05_08_right_slice=sr_out_02_05_08_right_slice.permute(0,2,1,3)
            
            sr_out_2_3_4_6_7_8_left=[]
            for i in range(sr_out_2_3_4_6_7_8_left_slice.shape[-1]):
                if (i%4!=0):
                    sr_out_2_3_4_6_7_8_left.append(sr_out_2_3_4_6_7_8_left_slice[:,:,:,i])
            sr_out_2_3_4_6_7_8_left=torch.stack(sr_out_2_3_4_6_7_8_left,dim=-1)
            
            sr_out_2_3_4_6_7_8_right=[]
            for i in range(sr_out_2_3_4_6_7_8_right_slice.shape[-1]):
                if (i%4!=0):
                    sr_out_2_3_4_6_7_8_right.append(sr_out_2_3_4_6_7_8_right_slice[:,:,:,i])
            sr_out_2_3_4_6_7_8_right=torch.stack(sr_out_2_3_4_6_7_8_right,dim=-1) 
            sr_out_02_05_08_left=[]
            for i in range(sr_out_02_05_08_left_slice.shape[-1]):
                if (i%4!=0):
                    sr_out_02_05_08_left.append(sr_out_02_05_08_left_slice[:,:,:,i])
            sr_out_02_05_08_left=torch.stack(sr_out_02_05_08_left,dim=-1)
            
            sr_out_02_05_08_right=[]
            for i in range(sr_out_02_05_08_right_slice.shape[-1]):
                if (i%4!=0):
                    sr_out_02_05_08_right.append(sr_out_02_05_08_right_slice[:,:,:,i])
            sr_out_02_05_08_right=torch.stack(sr_out_02_05_08_right,dim=-1)  
            
            return  interpolation_out,sr_left,sr_right,interpolation_out_2_3_4_6_7_8,sr_out_2_3_4_6_7_8_left,sr_out_2_3_4_6_7_8_right,m_items,interpolation_out_02_05_08,sr_out_02_05_08_left,sr_out_02_05_08_right, softmax_score_query,softmax_score_memory,separateness_loss,compactness_loss
            
