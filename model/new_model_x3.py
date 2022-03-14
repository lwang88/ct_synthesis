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
            slice1_10=lr_volume[:,:,:,:-1].permute(0,3,1,2).reshape(-1,1,c,k)#1_frame,10_frame
            slice10_19=lr_volume[:,:,:,1:].permute(0,3,1,2).reshape(-1,1,c,k)#10_frame,19_frame
            slice4_7_13_16,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(slice1_10,slice10_19,m_items,train)#3,7,11
            slice4_7_13_16 = slice4_7_13_16.contiguous().view(-1,1,c,k)
            slice1_4_7_10_13_16=input_volume[:,:,:,:-1].permute(0,3,1,2).reshape(-1,1,c,k)
            slice4_7_10_13_16_19=input_volume[:,:,:,1:].permute(0,3,1,2).reshape(-1,1,c,k)
            slice2_3_5_6,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(slice1_4_7_10_13_16,slice4_7_10_13_16_19,m_items,train)
            slice2_3_5_6 = slice2_3_5_6.contiguous().view(-1,1,c,k)
            slice1_2_3_4_5_6_7=[]
            slice2_3_4_5_6_7_8=[]
            for i in range(slice1_4_7_10_13_16.shape[0]):
                slice1_2_3_4_5_6_7.append(slice1_4_7_10_13_16[i,:,:,:])
                slice1_2_3_4_5_6_7.append(slice2_3_5_6[2*i,:,:,:])
                slice1_2_3_4_5_6_7.append(slice2_3_5_6[2*i+1,:,:,:])
            for i in range(slice4_7_10_13_16_19.shape[0]):
                slice2_3_4_5_6_7_8.append(slice2_3_5_6[2*i,:,:,:])
                slice2_3_4_5_6_7_8.append(slice2_3_5_6[2*i+1,:,:,:])
                slice2_3_4_5_6_7_8.append(slice4_7_10_13_16_19[i,:,:,:])
            slice1_2_3_4_5_6_7 = torch.cat(slice1_2_3_4_5_6_7,0).unsqueeze(1)
            slice2_3_4_5_6_7_8 = torch.cat(slice2_3_4_5_6_7_8,0).unsqueeze(1)
            slice03_06,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(slice1_2_3_4_5_6_7,slice2_3_4_5_6_7_8,m_items,train) #1+2->03,07
            slice03_06 = slice03_06.contiguous().view(-1,1,c,k)
            
            interpolation_out=slice4_7_13_16.contiguous().view(b,-1,c,k).permute(0,2,3,1)
            interpolation_out_2_3_5_6_8_9=slice2_3_5_6.reshape(b,-1,c,k).permute(0,2,3,1)
            interpolation_out_03_06=slice03_06.reshape(b,-1,c,k).permute(0,2,3,1)
            
            sr_lr_volume = lr_volume.unsqueeze(1).permute(0,2,1,3,4).reshape(-1,1,k,g)#1,10,19
            sr_out = self.model1(sr_lr_volume)#1,4,7,10,13,16,19,21,23
            sr_out =sr_out[:,:,:,:-2]#1,4,7,10,13,16,19
            b1,_,_,g1=sr_out.shape
            sr_left=sr_out.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,k,-1).permute(0,2,1,3,4).squeeze(1)
            
            sr_lr_volume = lr_volume.unsqueeze(1).permute(0,3,1,2,4).reshape(-1,1,c,g)#1,10,19
            sr_out = self.model1(sr_lr_volume)#1,4,7,10,13,16,19,21,23
            sr_out =sr_out[:,:,:,:-2]#1,4,7,10,13,16,19
            b1,_,_,g1=sr_out.shape
            sr_right=sr_out.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,c,-1).permute(0,2,1,3,4).squeeze(1).permute(0,2,1,3)
            
            sr_out_left=input_volume
            sr_out_retrain_left = sr_out_left.unsqueeze(1).permute(0,2,1,3,4).contiguous().view(-1,1,k,g)
            sr_out_retrain_right = sr_out_left.unsqueeze(1).permute(0,3,1,2,4).contiguous().view(-1,1,c,g)
            sr_out_retrain_left_right = torch.cat((sr_out_retrain_left,sr_out_retrain_right),0)
            
            sr_out_2_3_5_6_8_9_left_right = self.model1(sr_out_retrain_left_right)[:,:,:,:-2]#pici,chang,kuang,gao,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21
            sr_out_03_06_left_right = self.model1(sr_out_2_3_5_6_8_9_left_right)[:,:,:,:-2]
            
            sr_out_2_3_5_6_8_9_left_slice,sr_out_2_3_5_6_8_9_right_slice = sr_out_2_3_5_6_8_9_left_right.chunk(2,dim=0)
            sr_out_03_06_left_slice,sr_out_03_06_right_slice = sr_out_03_06_left_right.chunk(2,dim=0)
            
            sr_out_2_3_5_6_8_9_left_slice=sr_out_2_3_5_6_8_9_left_slice.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,c,-1).permute(0,2,1,3,4).squeeze(1)
            sr_out_2_3_5_6_8_9_right_slice=sr_out_2_3_5_6_8_9_right_slice.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,k,-1).permute(0,2,1,3,4).squeeze(1)
            sr_out_03_06_left_slice=sr_out_03_06_left_slice.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,c,-1).permute(0,2,1,3,4).squeeze(1)
            sr_out_03_06_right_slice=sr_out_03_06_right_slice.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,k,-1).permute(0,2,1,3,4).squeeze(1)
            sr_out_2_3_5_6_8_9_right_slice=sr_out_2_3_5_6_8_9_right_slice.permute(0,2,1,3)
            sr_out_03_06_right_slice=sr_out_03_06_right_slice.permute(0,2,1,3)
            
            sr_out_2_3_5_6_8_9_left=[]
            for i in range(sr_out_2_3_5_6_8_9_left_slice.shape[-1]):
                if (i%3!=0):
                    sr_out_2_3_5_6_8_9_left.append(sr_out_2_3_5_6_8_9_left_slice[:,:,:,i])
            sr_out_2_3_5_6_8_9_left=torch.stack(sr_out_2_3_5_6_8_9_left,dim=-1)
            
            sr_out_2_3_5_6_8_9_right=[]
            for i in range(sr_out_2_3_5_6_8_9_right_slice.shape[-1]):
                if (i%3!=0):
                    sr_out_2_3_5_6_8_9_right.append(sr_out_2_3_5_6_8_9_right_slice[:,:,:,i])
            sr_out_2_3_5_6_8_9_right=torch.stack(sr_out_2_3_5_6_8_9_right,dim=-1) 
                
            sr_out_03_06_left=[]
            for i in range(sr_out_03_06_left_slice.shape[-1]):
                if (i%3!=0):
                    sr_out_03_06_left.append(sr_out_03_06_left_slice[:,:,:,i])
            sr_out_03_06_left=torch.stack(sr_out_03_06_left,dim=-1)
            
            sr_out_03_06_right=[]
            for i in range(sr_out_03_06_right_slice.shape[-1]):
                if (i%3!=0):
                    sr_out_03_06_right.append(sr_out_03_06_right_slice[:,:,:,i])
            sr_out_03_06_right=torch.stack(sr_out_03_06_right,dim=-1)     
            
            return  interpolation_out,sr_left,sr_right,interpolation_out_2_3_5_6_8_9,sr_out_2_3_5_6_8_9_left,sr_out_2_3_5_6_8_9_right,m_items,interpolation_out_03_06,sr_out_03_06_left,sr_out_03_06_right, softmax_score_query,softmax_score_memory,separateness_loss,compactness_loss
            
