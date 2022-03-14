# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:26:20 2022

@author: wl255
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter
plt.rcParams['text.usetex']=True
plt.rcParams['xtick.labelsize'] = 15 
plt.rcParams['ytick.labelsize'] = 15 
font_axes = {'family': 'Times New Roman',
                 #'weight': 'normal',
                 'size': 16,
              'color'  : 'black'
                 }

y=[0.9313,0.9404,0.9375,0.9342]
x = ["10%","25%","50%","100%"] 
total_width, n = 0.8, 4    
width = total_width / n
plt.grid(True, axis='y',color="#373e02", linestyle='--',alpha=0.5)
plt.bar(x, y,width=width,linestyle='-',edgecolor='black', lw=1)
plt.xlabel(r'$\gamma$',font_axes)
plt.ylabel("SSIM",font_axes)
plt.ylim((0.93, 0.9410))
plt.savefig(r'./ssim.pdf',transparent=True,pad_inches=0.1,bbox_inches = 'tight')
"""
y=[40.32,41.11,40.73,40.46]
x = ["10","25","50","100"]
total_width, n = 0.8, 4    
width = total_width / n
plt.grid(True, axis='y',color="#373e02", linestyle='--',alpha=0.5)
plt.bar(x, y,width=width,linestyle='-',edgecolor='black', lw=1) 
plt.xlabel(r'$\gamma$',font_axes)
plt.ylabel("PSNR",font_axes) 
plt.ylim((40.0, 41.2))
plt.savefig(r'./psnr.pdf',transparent=True,pad_inches=0.1,bbox_inches = 'tight')
"""