  # -*- coding: utf-8 -*-

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from torch.nn import init
from torch.optim import Adam      
from data.data import *
from torch.utils.tensorboard import SummaryWriter
os.environ['CUDA_VISIBLE_DEVICES']="0"
import config
args, unparsed = config.get_args()
from PIL import Image 
from scipy.signal import convolve2d
from helper.ssim import SSIM as ssim_
compute_ssim=ssim_().cuda(0)

if args.upscale==2:
    from test_help.test_model import parallel_model as model 
    model = model(args)
    
elif args.upscale==3:
    from test_help.testx3_model import parallel_model as model 
    model = model(args) 
       
elif args.upscale==4:
    from test_help.testx4_model import parallel_model as model 
    model = model(args)    

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
     
def testx2(args):
    model.cuda(0)
    m_items=torch.load(r"/home/liangwang/Desktop/cvpr_ct_sr/checkpoint/memory_items/x2/"+"4keys.pt").cuda(0)
    pretrained_dict = torch.load(r"/home/liangwang/Desktop/cvpr_ct_sr/checkpoint/logs/x2/"+"4_model.pk")
    model.model1.load_state_dict((pretrained_dict["sr"]))
    model.model2.load_state_dict((pretrained_dict["interpolation"]))
    model.eval()
    L1_lossfunction = set_loss(args)
    L1_lossfunction = L1_lossfunction.cuda(0)    
    test_loader = testdata(r"/home/liangwang/Desktop/ct_data/test_data_64/")
    dataloader = torch.utils.data.DataLoader(test_loader, batch_size=args.batch_size,
    drop_last=True, shuffle=False, num_workers=7, pin_memory=False)
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
        with torch.no_grad():
            interpolation_out_2 = model(lr_volume,m_items,train=False)#1,2,3,4,5,6,7,8,9
        out_volume = interpolation_out_2.clip(0,1).squeeze(0)
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
       
def testx3(args):
    model.cuda(0)
    m_items=torch.load(r"/home/liangwang/Desktop/cvpr_ct_sr/checkpoint/memory_items/x3/"+"52keys.pt").cuda(0)
    pretrained_dict = torch.load(r"/home/liangwang/Desktop/cvpr_ct_sr/checkpoint/logs/x3/"+"52_model.pk")
    model.model1.load_state_dict((pretrained_dict["interpolation"]))
    model.model2.load_state_dict((pretrained_dict["sr"]))
    model.eval()
    L1_lossfunction = set_loss(args)
    L1_lossfunction = L1_lossfunction.cuda(0)    
    test_loader = testdata(r"/home/liangwang/Desktop/ct_data/test_data_64/")
    dataloader = torch.utils.data.DataLoader(test_loader, batch_size=1,
    drop_last=True, shuffle=False, num_workers=7, pin_memory=False) 
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
        while( (input_volume.shape[3]-1)%3!=0 ):#1,4,7,10,13
            input_volume=input_volume[:,:,:,:-1].float()
        gt = input_volume.squeeze().cuda(0) 
        lr_volume = input_volume[:,:,:,::3].float().cuda(0)#1,512,512,24(1,3,5,7..n)
        with torch.no_grad():
            interpolation_out_2_3 = model(lr_volume.detach(),m_items.detach(),train=False)
        out_volume = interpolation_out_2_3.clip(0,1).cuda(0).squeeze(0)
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
   # ct_volume =interpolation_out.numpy().clip(0,1)
   # hr_volume=nib.Nifti1Image(ct_volume,np.eye(4))
   # hr_path=os.path.join(r"/home/liangwang/Desktop/ct_interpolation_super_resolution/volume_view/","ctx3saint.nii.gz")
   # nib.save(hr_volume,hr_path) 
   
def testx4(args):
    model.cuda(0)
    m_items=torch.load(r"/home/liangwang/Desktop/cvpr_ct_sr/checkpoint/memory_items/x4/"+"48keys.pt").cuda(0)
    pretrained_dict = torch.load(r"/home/liangwang/Desktop/cvpr_ct_sr/checkpoint/logs/x4/"+"48_model.pk")
    model.model1.load_state_dict((pretrained_dict["interpolation"]))
    model.model2.load_state_dict((pretrained_dict["sr"]))
    model.eval()
    L1_lossfunction = set_loss(args)
    test_loader = testdata(r"/home/liangwang/Desktop/ct_data/test_data_64/")
    dataloader = torch.utils.data.DataLoader(test_loader, batch_size=1, 
    drop_last=True, shuffle=False, num_workers=7, pin_memory=False)
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
        while((input_volume.shape[3]-1)%4!=0 ):#1,5,9,13,17
            input_volume=input_volume[:,:,:,:-1].float()
        gt = input_volume.squeeze().cuda(0)
        lr_volume = input_volume[:,:,:,::4].float().cuda(0)
        with torch.no_grad():
            interpolation_out_2 = model(lr_volume.detach(),m_items.detach())
        out_volume = interpolation_out_2.clip(0,1).squeeze().cuda(0)
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
        if psnr<0:
            print(gt.max(),gt.min(),out_volume.max(),out_volume.min())
        log = r"[{} / {}]    PSNR: {:.4f}       x_y_ssim: {:.4f}    x_z_ssim: {:.4f}    y_z_ssim: {:.4f} ".format(i+1,len(test_loader),psnr,x_y_ssim, x_z_ssim,    y_z_ssim)
        print(log)
        total_x_y_ssim+=x_y_ssim
        total_x_z_ssim+=x_z_ssim
        total_y_z_ssim+=y_z_ssim
    print("3_2_4average_psnr:",average_psnr/(i+1)) 
    print("average_x_y_ssim:",total_x_y_ssim/(i+1)) 
    print("average_x_z_ssim:",total_x_z_ssim/(i+1)) 
    print("average_y_z_ssim:",total_y_z_ssim/(i+1)) 
  #  ct_volume =out_volume.clip(0,1).cpu().detach().numpy() * 2442 - 1024 
  #  hr_volume=nib.Nifti1Image(ct_volume,np.eye(4))
  #  hr_volume.header["pixdim"][3]
  #  hr_path=os.path.join(r"/home/liangwang/Desktop/ct_interpolation_super_resolution/volume_view/","x4cthep.nii.gz")
  #  nib.save(hr_volume,hr_path) 
 
if __name__ == '__main__':
    if args.upscale==2:
        testx2(args)
    elif args.upscale==3:
        testx3(args)  
    elif args.upscale==4:
        testx4(args)
