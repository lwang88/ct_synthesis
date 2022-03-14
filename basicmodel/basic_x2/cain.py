import math
import numpy as np

import torch
import torch.nn as nn

from basicmodel.common import *
from basicmodel.memory_ranking import *



class Encoder(nn.Module):
    def __init__(self, in_channels=3, depth=3):
        super(Encoder, self).__init__()
        # Shuffle pixels to expand in channel dimension
        # shuffler_list = [PixelShuffle(0.5) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        
        self.shuffler = PixelShuffle(1 / 2**depth)
        relu = nn.LeakyReLU(0.2, True)
        # FF_RCAN or FF_Resblocks
        self.interpolate = Interpolation(12, in_channels * (4**depth), act=relu)
        
    def forward(self, x1, x2):
        """
        Encoder: Shuffle-spread --> Feature Fusion --> Return fused features
        """
        feats1 = self.shuffler(x1)
        feats2 = self.shuffler(x2)

        feats,res2,res1 = self.interpolate(feats1, feats2)
        return feats,res2,res1


class Decoder(nn.Module):
    def __init__(self, in_channels=3,depth=3):
        super(Decoder, self).__init__()
        relu = nn.LeakyReLU(0.2, True)
        # shuffler_list = [PixelShuffle(2) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = PixelShuffle(2**3)
        self.interpolate_decoder= Interpolation_decoder( 12, in_channels * (4**3), act=relu)
    def forward(self, updated_feats,x2,x1):
        feats = self.interpolate_decoder(updated_feats,x2,x1)
        out = self.shuffler(feats)
        return out

class CAIN(nn.Module):
    def __init__(self, depth=3):
        super(CAIN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=1,padding=0)
        self.conv2 = nn.Conv2d(in_channels=3,out_channels=1,kernel_size=1,padding=0)
        self.encoder = Encoder(in_channels=3, depth=3)
        self.decoder = Decoder(depth=depth)
        self.memory = Memory(memory_size=10,feature_dim=192, key_dim=192, temp_update=0.1, temp_gather=0.1)
    def forward(self, x1, x2,keys,train=True):
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)
        fea,encoder_2,encoder_1 = self.encoder(x1, x2)
        if train:
            updated_fea, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = self.memory(fea, keys, train)
            out = self.decoder(updated_fea,encoder_2,encoder_1)
            out = self.conv2(out)
            return out, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss
        #test
        else:
            updated_fea, keys, softmax_score_query, softmax_score_memory,query, top1_keys, keys_ind, compactness_loss = self.memory(fea, keys, train)
            out = self.decoder(updated_fea,encoder_2,encoder_1)
            out = self.conv2(out)
            return out, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss  
