# -*- coding: utf-8 -*-
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
            slice1_5_9=lr_volume[:,:,:,:-1].permute(0,3,1,2).reshape(-1,1,c,k)#1_frame,5_frame,9_frame
            slice5_9_13=lr_volume[:,:,:,1:].permute(0,3,1,2).reshape(-1,1,c,k)#5_frame,9_frame,13_frame
            slice3_7_11,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(slice1_5_9,slice5_9_13,m_items,train)#3,7,11
            slice1_3_5_7_9_11=input_volume[:,:,:,:-1].permute(0,3,1,2).reshape(-1,1,c,k)
            slice3_5_7_9_11_13=input_volume[:,:,:,1:].permute(0,3,1,2).reshape(-1,1,c,k)
            slice2_6_10_4_8_12,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = self.model2(slice1_3_5_7_9_11,slice3_5_7_9_11_13,m_items,train)
            slice1_2_3_4_5_6_7_8_9_10_11_12 = []
            slice2_3_4_5_6_7_8_9_10_11_12_13 = []
            for i in range(slice1_3_5_7_9_11.shape[0]):
                slice1_2_3_4_5_6_7_8_9_10_11_12.append(slice1_3_5_7_9_11[i,:,:,:])
                slice1_2_3_4_5_6_7_8_9_10_11_12.append(slice2_6_10_4_8_12[i,:,:,:])
            for i in range(slice2_6_10_4_8_12.shape[0]):
                slice2_3_4_5_6_7_8_9_10_11_12_13.append(slice2_6_10_4_8_12[i,:,:,:])
                slice2_3_4_5_6_7_8_9_10_11_12_13.append(slice3_5_7_9_11_13[i,:,:,:])
            slice1_2_3_4_5_6_7_8_9_10_11_12=torch.cat(slice1_2_3_4_5_6_7_8_9_10_11_12,dim=0).unsqueeze(1)
            slice2_3_4_5_6_7_8_9_10_11_12_13=torch.cat(slice2_3_4_5_6_7_8_9_10_11_12_13,dim=0).unsqueeze(1)
            slice_05,  _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= self.model2(slice1_2_3_4_5_6_7_8_9_10_11_12,slice2_3_4_5_6_7_8_9_10_11_12_13,m_items,train)
            
            interpolation_out=slice3_7_11.reshape(b,-1,c,k).permute(0,2,3,1)#3,7,11
            interpolation_out_2_4=slice2_6_10_4_8_12.reshape(b,-1,c,k).permute(0,2,3,1)#2,4,6,8,10,12
            interpolation_out_05=slice_05.reshape(b,-1,c,k).permute(0,2,3,1)
            
            sr_lr_volume = lr_volume.unsqueeze(1).permute(0,2,1,3,4).view(-1,1,k,g)#1,5,9,13
            sr_out = self.model1(sr_lr_volume)#pici,chang,kuang,gao,1,3,5,7,9,11,13,15
            sr_out =sr_out[:,:,:,:-1]#8 to 7    1,3,5,7,9,11,13,
            b1,_,_,g1=sr_out.shape
            sr_left=sr_out.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,k,-1).permute(0,2,1,3,4).squeeze(1)
            
            sr_lr_volume = lr_volume.unsqueeze(1).permute(0,3,1,2,4).reshape(-1,1,c,g)#1,5,9,13
            sr_out = self.model1(sr_lr_volume)#pici,chang,kuang,gao,1,3,5,7,9,11,13,15
            sr_out =sr_out[:,:,:,:-1]#8 to 7    1,3,5,7,9,11,13,
            b1,_,_,g1=sr_out.shape
            sr_right=sr_out.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,k,-1).permute(0,2,1,3,4).squeeze(1).permute(0,2,1,3)
            
            sr_out_left=input_volume
            sr_out_retrain_left = sr_out_left.unsqueeze(1).permute(0,2,1,3,4).contiguous().view(-1,1,k,g1)
            sr_out_retrain_right = sr_out_left.unsqueeze(1).permute(0,3,1,2,4).contiguous().view(-1,1,c,g1)
            sr_out_retrain_left_right = torch.cat((sr_out_retrain_left,sr_out_retrain_right),0)
            sr_out_2_4_left_right = self.model1(sr_out_retrain_left_right)[:,:,:,:-1]#pici,chang,kuang,gao ,1,2,3,4,5,6,7,8,9,10,11,12,13,|\14
            sr_out_05_left_right = self.model1(sr_out_2_4_left_right)[:,:,:,:-1]#1,1.5,2,2.5,3,3.5.....13,|13.5
            
            sr_out_2_4_left,sr_out_2_4_right = sr_out_2_4_left_right.chunk(2,dim=0)
            sr_out_05_left,sr_out_05_right = sr_out_05_left_right.chunk(2,dim=0)
            
            sr_out_2_4_left = sr_out_2_4_left.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,k,-1).permute(0,2,1,3,4).squeeze(1)
            sr_out_2_4_right = sr_out_2_4_right.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,c,-1).permute(0,2,1,3,4).squeeze(1)
            sr_out_05_left = sr_out_05_left.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,k,-1).permute(0,2,1,3,4).squeeze(1)
            sr_out_05_right = sr_out_05_right.contiguous().view(self.args.batch_size,b1//self.args.batch_size,1,c,-1).permute(0,2,1,3,4).squeeze(1)
            sr_out_2_4_right = sr_out_2_4_right.permute(0,2,1,3) 
            sr_out_05_right = sr_out_05_right.permute(0,2,1,3)
            sr_out_2_4_left=sr_out_2_4_left[:,:,:,1:-1:2]
            sr_out_2_4_right=sr_out_2_4_right[:,:,:,1:-1:2]
            sr_out_05_left=sr_out_05_left[:,:,:,1:-1:2]
            sr_out_05_right=sr_out_05_right[:,:,:,1:-1:2]
            return  interpolation_out,sr_left,sr_right,interpolation_out_2_4,sr_out_2_4_left,sr_out_2_4_right,m_items,interpolation_out_05,sr_out_05_left,sr_out_05_right, softmax_score_query,softmax_score_memory,separateness_loss,compactness_loss
            
