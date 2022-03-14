# -*- coding: utf-8 -*-
import os
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from torch.nn import init
from torch.optim import Adam
from test_model import parallel_model as model        
from data import *
from torch.utils.tensorboard import SummaryWriter
os.environ['CUDA_VISIBLE_DEVICES']="3"
import config
args, unparsed = config.get_args()
model = model(args)
from PIL import Image 
from scipy.signal import convolve2d
from ssim import SSIM as ssim_
compute_ssim=ssim_().cuda(0)


def set_loss(args):
    lossType = args.loss
    if lossType == 'MSE':
        lossfunction = nn.MSELoss()
    elif lossType == 'L1':
        lossfunction = nn.L1Loss()
    return lossfunction
    
def calc_psnr(img1, img2,max=255.0):
    mse = ((img1-img2)**2).mean()
    return 10. * ((max**2)/(mse)).log10()
def calc_psnr_1(img1, img2,max=1.0):
    mse = ((img1-img2)**2).mean()
    return 10. * ((max**2)/(mse)).log10() 
    
     
def test(args):
    model.cuda(0)
    m_items=torch.load(r"/home/liangwang/Desktop/cvpr_ct_sr/checkpoint/memory_items/x2/"+"8keys.pt").cuda(0)
    pretrained_dict = torch.load(r"/home/liangwang/Desktop/cvpr_ct_sr/checkpoint/logs/x2/"+"8_model.pk")
    model.model1.load_state_dict((pretrained_dict["interpolation"]))
    model.model2.load_state_dict((pretrained_dict["sr"]))
    model.eval()
    L1_lossfunction = set_loss(args)
    L1_lossfunction = L1_lossfunction.cuda(0)    
    test_loader = testdata(r"/home/liangwang/Desktop/ct_data/test_data/")
    dataloader = torch.utils.data.DataLoader(test_loader, batch_size=args.batch_size,
    drop_last=True, shuffle=True, num_workers=7, pin_memory=False)
    average_psnr=0
    total_x_y_ssim=0
    total_x_z_ssim=0
    total_y_z_ssim=0
    for  i, input_volume in enumerate(dataloader):
        psnr=0  
        loss=0
        x_y_ssim=0 
        x_z_ssim=0
        y_z_ssim=0
        ssim=0
        if input_volume.shape[3]%2==0:
            input_volume=input_volume[:,:,:,:-1].float()#1,2,3,4,5,6,7,8,9,10,11,12,13
        gt = input_volume.squeeze().cuda(0).float()
        lr_volume = input_volume[:,:,:,::2].float().cuda(0)#1,512,512,24(1,3,5,7..n)
        interpolation_out_2 ,  _= model(lr_volume,m_items,train=False)#1,2,3,4,5,6,7,8,9
        out_volume = interpolation_out.clip(0,1).squeeze(0)
        psnr = calc_psnr_1(out_volume,gt).item()
        average_psnr+=psnr
        for j in range(gt.shape[2]):
            ssim=compute_ssim(gt[:,:,j].double(),out_volume[:,:,j].double())
            x_y_ssim+=ssim.item()
        x_y_ssim=x_y_ssim/(j+1)
        for t in range(gt.shape[0]):
            ssim=compute_ssim(gt[t,:,:].double(),out_volume[t,:,:].double())
            x_z_ssim+=ssim.item()
        x_z_ssim=x_z_ssim/(t+1)
        for q in range(gt.shape[1]):
            ssim=compute_ssim(gt[:,q,:].double(),out_volume[:,q,:].double())
            y_z_ssim+=ssim.item()
        y_z_ssim=y_z_ssim/(q+1)
        log = r"[{} / {}]    PSNR: {:.4f}       x_y_ssim: {:.4f}    x_z_ssim: {:.4f}    y_z_ssim: {:.4f} ".format(i+1,len(test_loader),psnr,x_y_ssim, x_z_ssim,    y_z_ssim)
        print(log)
        total_x_y_ssim+=x_y_ssim
        total_x_z_ssim+=x_z_ssim
        total_y_z_ssim+=y_z_ssim
    print("3_2_4average_psnr:",average_psnr/(i+1)) 
    print("average_x_y_ssim:",total_x_y_ssim/(i+1)) 
    print("average_x_z_ssim:",total_x_z_ssim/(i+1)) 
    print("average_y_z_ssim:",total_y_z_ssim/(i+1)) 
    #ct_volume =interpolation_out_2.clip(0,1).cpu().detach().numpy()     
    #hr_volume=nib.Nifti1Image(ct_volume,np.eye(4))
    #hr_path=os.path.join(r"/home/liangwang/Desktop/ct_interpolation_super_resolution/volume_view/","ctx2rdn.nii.gz")
  #  nib.save(hr_volume,hr_path)
       

 
if __name__ == '__main__':
    test(args)

