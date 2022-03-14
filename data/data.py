import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import nibabel as nib
import random 
import cv2 
import config
args, unparsed = config.get_args()


class volumedata(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.trainlist=os.listdir(self.data_root)
        random.shuffle(self.trainlist)
        self.trainlist = self.trainlist[::5]
         
    def __getitem__(self, index):
        volumepath = os.path.join(self.data_root, self.trainlist[index])
        volume=nib.load(volumepath)
        volumeIn=np.array(volume.get_fdata()).astype(float)#128,128,13
        if args.data_type!="direct":
            volumeIn=volumeIn.transpose(2,0,1)
            volumeIn = cv2.GaussianBlur(volumeIn, (3,3), 1.3)
            volumeIn=volumeIn+np.random.normal(0, 25, volumeIn.shape)
            volumeIn=volumeIn.transpose(1,2,0)
        if random.random() >= 0.5:
            volumeIn = volumeIn[:,::-1,:].copy()
        if random.random() >= 0.5:
            volumeIn = volumeIn[::-1,:,:].copy()
        ma,mi=3072.0 , -1024.0
        if (ma - mi) !=0:
            volumeIn = (volumeIn - mi)/(ma - mi)
        volumeIn=torch.from_numpy(volumeIn)
        return volumeIn

    def __len__(self):
        return len(self.trainlist)
       

class testdata(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.trainlist = os.listdir(self.data_root)#[::12]

    def __getitem__(self, index):
        imgpath = os.path.join(self.data_root, self.trainlist[index])
        imgIn=nib.load(imgpath)
        imgIn=np.array([imgIn.get_fdata()]).astype(float)
        imgIn=imgIn.squeeze()#512,512,48
        ma,mi=3072.0 , -1024.0
        imgIn = (imgIn - mi)/(ma - mi)
        imgIn=torch.from_numpy(imgIn)
        return imgIn

    def __len__(self):
        return len(self.trainlist)
