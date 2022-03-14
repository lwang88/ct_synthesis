  # -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from torch.nn import init
from torch.optim import Adam
from pytorch_wavelets import DWTForward, DWTInverse    
from data.data import *
from torch.utils.tensorboard import SummaryWriter
os.environ['CUDA_VISIBLE_DEVICES']="0"
import config
args, unparsed = config.get_args()
loss_activ=torch.nn.ReLU()
loss_=torch.nn.MSELoss(reduce=False, reduction='None' )

if args.upscale==2:
    from model.new_model_x2 import parallel_model as model 
    model = model(args)
elif args.upscale==3:
    from model.new_model_x3 import parallel_model as model 
    model = model(args)    
elif args.upscale==4:
    from model.new_model_x4 import parallel_model as model 
    model = model(args)    

optimizer = Adam(model.parameters(),lr=args.lr*0.1, betas=(args.beta1, args.beta2))

def adjust_learning_rate(optimizer, epoch, lr):
    if epoch>60:
        lr *= (0.5 * ((epoch-60) // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class MyMSELoss(torch.nn.Module):
    def __init__(self):
        super(MyMSELoss, self).__init__()
        
    def forward(self, output, groundtruth,k):
        loss=torch.pow((output - groundtruth),2)
        loss,_ = torch.topk(loss.flatten(),k,largest=False)
        return torch.mean(loss)

class xiaobobianhuan(torch.nn.Module):
    def __init__(self):
        super(xiaobobianhuan, self).__init__()
        
    def forward(self,slice3, groundtruth,):
        slice3=slice3.permute(0,3,1,2)
        groundtruth=groundtruth.permute(0,3,1,2)
        xfm = DWTForward(J=3, wave='db3', mode='zero').cuda(0)
        out_slice3,Yh_out=xfm(slice3.cuda(0))
        out_groundtruth,Yh_gd=xfm(groundtruth.cuda(0))
        loss1=torch.pow((Yh_out[0] - Yh_gd[0]),2)
        loss2=torch.pow((Yh_out[1] - Yh_gd[1]),2)
        loss3=torch.pow((Yh_out[2] - Yh_gd[2]),2)
        return torch.mean(loss1) +torch.mean(loss2) +torch.mean(loss3) 

def set_loss(args):
    lossType = args.loss
    if lossType == 'MSE':
        lossfunction = nn.MSELoss()
    elif lossType == 'L1':
        lossfunction = nn.L1Loss()
    return lossfunction
    
def calc_psnr_1(img1, img2,max=1.0):
    mse = ((img1-img2)**2).mean()
    return 10. * ((max**2)/(mse)).log10()
      

def train(args): 
    model.cuda(0)
    myloss = MyMSELoss().cuda(0)
    xiaobo_loss=xiaobobianhuan().cuda(0)
    m_items = F.normalize(torch.rand((10, 192), dtype=torch.float), dim=1).cuda(0) # Initialize the memory items
    L1_lossfunction = set_loss(args)
    L1_lossfunction = L1_lossfunction.cuda(0)
    if args.upscale==2:
        if args.resume:
            m_items=torch.load(r"/home/liangwang/Desktop/cvpr_ct_sr/checkpoint/memory_items/x2/"+"4keys.pt").cuda(0)
            pretrained_dict = torch.load(r"/home/liangwang/Desktop/cvpr_ct_sr/checkpoint/logs/x2/"+"4_model.pk")
            model.model1.load_state_dict((pretrained_dict["sr"]))
            model.model2.load_state_dict((pretrained_dict["interpolation"]))
            print("resume")
        for epoch in range(args.max_epoch):
            train_loader = volumedata(r"/home/liangwang/Desktop/ct_data/train_data_64_1_2_3_33/")
            dataloader = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size,\
        drop_last=True, shuffle=True, num_workers=32, pin_memory=False)
            PSNR =0
            psnr =0
            loss_total=0
            loss_interpolation=0
            loss_rdn=0
            loss_consistency=0
            loss_1 =0
            for  i, input_volume in enumerate(dataloader): 
                t = random.randint(0,15)
                input_volume = input_volume[:,:,:,t:t+13]
                groundtruth_volume=input_volume[:,:,:,::2].float().cuda(0)#1,3,5,7,9,11,13
                groundtruth=groundtruth_volume[:,:,:,1:-1:2]#3,7,11
                optimizer.zero_grad()
                interpolation_out,sr_out_left,sr_out_right,interpolation_out_2_4,sr_out_2_4_left,sr_out_2_4_right,m_items,interpolation_out_05,sr_out_05_left,sr_out_05_right,softmax_score_query,softmax_score_memory,separateness_loss,compactness_loss= model(input_volume.float().cuda(0),m_items,train=True)
                loss_interpolation=L1_lossfunction(groundtruth,interpolation_out)
                loss_sr_left=L1_lossfunction(groundtruth_volume,sr_out_left)
                loss_sr_right=L1_lossfunction(groundtruth_volume,sr_out_right)
                xiaobo_inter=xiaobo_loss(groundtruth,interpolation_out)
                xiaobo_sr_left=xiaobo_loss(groundtruth_volume,sr_out_left)
                xiaobo_sr_right=xiaobo_loss(groundtruth_volume,sr_out_right)
                loss_consistency_right=myloss(interpolation_out_2_4,sr_out_2_4_right,5000)#the smallest 5000 
                loss_consistency_left=myloss(interpolation_out_2_4,sr_out_2_4_left,5000)
                loss_consistency_right_05=myloss(interpolation_out_05,sr_out_05_right,2000)#the smallest 2000 
                loss_consistency_left_05=myloss(interpolation_out_05,sr_out_05_left,2000)
                loss_total=loss_interpolation+0.1 * compactness_loss + 0.1 * separateness_loss + loss_sr_left+loss_sr_right  + 0.15*(loss_consistency_right + loss_consistency_left + loss_consistency_left_05 + loss_consistency_right_05)+0.1*xiaobo_inter+0.1*(xiaobo_sr_left+xiaobo_sr_right)
                loss_total.backward()
                optimizer.step()
                adjust_learning_rate(optimizer,epoch,optimizer.param_groups[0]['lr'])
                loss_1 += loss_total.item()  
                psnr = calc_psnr_1(interpolation_out,groundtruth)
                PSNR +=psnr.item()
            if  (epoch+1) % 2 == 0:
                log = r"[{} / {}]   PSNR: {:.4f}     L1_loss: {:.4f}   number {}".format(epoch+1, args.max_epoch, PSNR/(int(i)+1),loss_1,(int(i)+1)*args.batch_size)
                print(log)
                if  (epoch+1) % 2 == 0:
                    memory_items=m_items.cpu()
                    torch.save(memory_items, os.path.join(r"./checkpoint/memory_items/x2",str(epoch+1)+ 'keys.pt'))
                    state = {'sr':model.model1.state_dict(),'interpolation':model.model2.state_dict()}
                    torch.save(state, os.path.join(r"./checkpoint/logs/x2",str(epoch+1)+ '_model.pk'))
    elif args.upscale==3:
        if args.resume:
            m_items=torch.load(r"/home/liangwang/Desktop/cvpr_ct_sr/checkpoint/memory_items/x3/"+"52keys.pt").cuda(0)
            pretrained_dict = torch.load(r"/home/liangwang/Desktop/cvpr_ct_sr/checkpoint/logs/x3/"+"52_model.pk")
            model.model1.load_state_dict((pretrained_dict["sr"]))
            model.model2.load_state_dict((pretrained_dict["interpolation"]))
            print("resume")
        for epoch in range(args.max_epoch):
            train_loader = volumedata(r"/home/liangwang/Desktop/ct_data/train_data_64_1_2_3_33/")
            dataloader = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size,\
        drop_last=True, shuffle=True, num_workers=32, pin_memory=False)
            PSNR =0
            psnr =0
            loss_total=0
            loss_interpolation=0
            loss_rdn=0
            loss_consistency=0
            loss_1 =0
            for  i, input_volume in enumerate(dataloader):
                groundtruth_volume=input_volume[:,:,:,::3].float().cuda(0)#1,4,7,10,13,16,19
                groundtruth=[]
                for j in range(groundtruth_volume.shape[3]):
                    if j%3!=0:
                        groundtruth.append(groundtruth_volume[:,:,:,j])
                groundtruth=torch.stack(groundtruth,3)#4,7,13,16
                optimizer.zero_grad()
                interpolation_out,sr_out_left,sr_out_right,interpolation_out_2_3_5_6_8_9,sr_out_2_3_5_6_8_9_left,sr_out_2_3_5_6_8_9_right,m_items,interpolation_out_03_06,sr_out_03_06_left,sr_out_03_06_right, softmax_score_query,softmax_score_memory,separateness_loss,compactness_loss= model(input_volume.float().cuda(0),m_items,train=True)
                loss_interpolation=L1_lossfunction(groundtruth,interpolation_out)
                loss_sr_left=L1_lossfunction(groundtruth_volume,sr_out_left)
                loss_sr_right=L1_lossfunction(groundtruth_volume,sr_out_right)
                xiaobo_inter=xiaobo_loss(groundtruth,interpolation_out)
                xiaobo_sr_left=xiaobo_loss(groundtruth_volume,sr_out_left)
                xiaobo_sr_right=xiaobo_loss(groundtruth_volume,sr_out_right)
                loss_consistency_right=myloss(interpolation_out_2_3_5_6_8_9,sr_out_2_3_5_6_8_9_left,5000)#the smallest 5000 
                loss_consistency_left=myloss(interpolation_out_2_3_5_6_8_9,sr_out_2_3_5_6_8_9_right,5000)
                loss_consistency_right_05=myloss(interpolation_out_03_06,sr_out_03_06_right,5000)#the smallest 5000 
                loss_consistency_left_05=myloss(interpolation_out_03_06,sr_out_03_06_left,5000)
                loss_total=loss_interpolation+0.1 * compactness_loss + 0.1 * separateness_loss + loss_sr_left+loss_sr_right  + 0.15*(loss_consistency_right + loss_consistency_left + loss_consistency_left_05 + loss_consistency_right_05)+0.1*xiaobo_inter+0.1*(xiaobo_sr_left+xiaobo_sr_right)
                loss_total.backward()
                optimizer.step()
                adjust_learning_rate(optimizer,epoch,optimizer.param_groups[0]['lr'])
                loss_1 += loss_total.item()  
                psnr = calc_psnr_1(interpolation_out,groundtruth)
                PSNR +=psnr.item()
            if  (epoch+1) % 2 == 0:
                log = r"[{} / {}]   PSNR: {:.4f}     L1_loss: {:.4f}   number {}".format(epoch+1, args.max_epoch, PSNR/(int(i)+1),loss_1,(int(i)+1)*args.batch_size)
                print(log)
                memory_items=m_items.cpu()
                torch.save(memory_items, os.path.join(r"./checkpoint/memory_items\x3",str(epoch+1)+ 'keys.pt'))
                state = {'sr':model.model1.state_dict(),'interpolation':model.model2.state_dict()}
                torch.save(state, os.path.join(r"./checkpoint/logs\x3",str(epoch+1)+ '_model.pk'))
    elif args.upscale==4:
        if args.resume:
            m_items=torch.load(r"/home/liangwang/Desktop/cvpr_ct_sr/checkpoint/memory_items/x4/"+"6keys.pt").cuda(0)
            pretrained_dict = torch.load(r"/home/liangwang/Desktop/cvpr_ct_sr/checkpoint/logs/x4/"+"6_model.pk")
            model.model1.load_state_dict((pretrained_dict["sr"]))
            model.model2.load_state_dict((pretrained_dict["interpolation"]))
            print("resume")
        for epoch in range(args.max_epoch):
            train_loader = volumedata(r"/home/liangwang/Desktop/ct_data/train_data_64_1_2_3_33/")
            dataloader = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size,\
        drop_last=True, shuffle=True, num_workers=32, pin_memory=False)
            PSNR =0
            psnr =0
            loss_total=0
            loss_interpolation=0
            loss_rdn=0
            loss_consistency=0
            loss_1 =0
            for  i, input_volume in enumerate(dataloader):
                groundtruth_volume=input_volume[:,:,:,::4].float().cuda(0)#1,5,9,13,17,21,25,29,33
                groundtruth=[]
                for j in range(groundtruth_volume.shape[3]):
                    if j%4!=0:
                        groundtruth.append(groundtruth_volume[:,:,:,j])
                groundtruth=torch.stack(groundtruth,3)
                optimizer.zero_grad()
                interpolation_out,sr_out_left,sr_out_right,interpolation_out_2_3_4_6_7_8_10_11_12_14_15_16,sr_out_2_3_4_6_7_8_10_11_12_14_15_16_left,sr_out_2_3_4_6_7_8_10_11_12_14_15_16_right,m_items,interpolation_out_02_05_08,sr_out_02_05_08_left,sr_out_02_05_08_right, softmax_score_query,softmax_score_memory,separateness_loss,compactness_loss= model(input_volume.float().cuda(0),m_items,train=True)
               # print(interpolation_out_2_3_4_6_7_8_10_11_12_14_15_16.shape)
               # print(sr_out_2_3_4_6_7_8_10_11_12_14_15_16_left.shape)
                loss_interpolation=L1_lossfunction(groundtruth,interpolation_out)
                loss_sr_left=L1_lossfunction(groundtruth_volume,sr_out_left)
                loss_sr_right=L1_lossfunction(groundtruth_volume,sr_out_right)
                xiaobo_inter=xiaobo_loss(groundtruth,interpolation_out)
                xiaobo_sr_left=xiaobo_loss(groundtruth_volume,sr_out_left)
                xiaobo_sr_right=xiaobo_loss(groundtruth_volume,sr_out_right)
                loss_consistency_right=myloss(interpolation_out_2_3_4_6_7_8_10_11_12_14_15_16,sr_out_2_3_4_6_7_8_10_11_12_14_15_16_left,5000)#the smallest 5000 
                loss_consistency_left=myloss(interpolation_out_2_3_4_6_7_8_10_11_12_14_15_16,sr_out_2_3_4_6_7_8_10_11_12_14_15_16_right,5000)
                loss_consistency_right_05=myloss(interpolation_out_02_05_08,sr_out_02_05_08_right,5000)#the smallest 5000 
                loss_consistency_left_05=myloss(interpolation_out_02_05_08,sr_out_02_05_08_left,5000)
                loss_total=loss_interpolation+0.1 * compactness_loss + 0.1 * separateness_loss + loss_sr_left+loss_sr_right  + 0.15*(loss_consistency_right + loss_consistency_left + loss_consistency_left_05 + loss_consistency_right_05)+0.1*xiaobo_inter+0.1*(xiaobo_sr_left+xiaobo_sr_right)
                loss_total.backward()
                adjust_learning_rate(optimizer,epoch,optimizer.param_groups[0]['lr'])
                optimizer.step()
                loss_1 += loss_total.item()  
                psnr = calc_psnr_1(interpolation_out,groundtruth)
                PSNR +=psnr.item()
            if  (epoch+1) % 2 == 0:
                log = r"[{} / {}]   PSNR: {:.4f}     L1_loss: {:.4f}   number {}".format(epoch+1, args.max_epoch, PSNR/(int(i)+1),loss_1,(int(i)+1)*args.batch_size)
                print(log)
                memory_items=m_items.cpu()
                if  (epoch+1) % 2 == 0:
                    torch.save(memory_items, os.path.join(r"./checkpoint/memory_items/x4",str(epoch+1)+ 'keys.pt'))
                    state = {'sr':model.model1.state_dict(),'interpolation':model.model2.state_dict()}
                    torch.save(state, os.path.join(r"./checkpoint/logs/x4",str(epoch+1)+ '_model.pk'))

if __name__ == '__main__':
    from setproctitle import setproctitle
    setproctitle("wangliang")
    train(args)
            
