# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:05:50 2020

@author: wangliang
"""
import numpy as np
import os
from os import path as op
import nibabel as nib


hr_data_path=r"/home/liangwang/Desktop/mri_data/train120_1_3_5/"
hr_filenames=os.listdir(hr_data_path)

    
lr_data_path=r"/home/liangwang/Desktop/mri_data/train_1_3_5_15/"

print(len(hr_filenames))

n=0
for filename in range(len(hr_filenames)):
    k=0
    hr_image_path=op.join(hr_data_path,hr_filenames[filename])
    hr_img=nib.load(hr_image_path)
    hrimg=np.asarray(hr_img.get_fdata())
    #hrimg=hrimg[:,:,:23]
    #print(lrimg.shape)
    for j in range(int(hrimg.shape[2]/15)):
        im_hr=hrimg[:,:,j*15:(j+1)*15,:]
        print(im_hr.shape)
        im_hr=nib.Nifti1Image(im_hr,np.eye(4))
        hr_image_path=str(k)+hr_filenames[filename]
        hr_new_path=op.join(lr_data_path,hr_image_path)
        nib.save(im_hr,hr_new_path)
        k=k+1
        n=n+1

print("n=",n)

    
