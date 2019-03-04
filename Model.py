# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:29:33 2019

@author: ericl
"""



import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from tqdm import tqdm, tqdm_notebook

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(11*129, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 3072),
            nn.Sigmoid(),
            nn.Linear(3072,2048))
        
            #nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            #nn.Linear(3, 12),
            #nn.ReLU(True),
            #nn.Linear(12, 64),
            #nn.ReLU(True),
            nn.Linear(2048,3072),
            nn.Linear(3072, 2048),
            nn.Linear(2048, 11*129))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class NetWork(nn.Module):
    def __init__(self):
        super(NetWork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(11*129, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Linear(512,256))
        
            #nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.last = nn.Sequential(
            #nn.Linear(3, 12),
            #nn.ReLU(True),
            #nn.Linear(12, 64),
            #nn.ReLU(True),
            #nn.Sigmoid(),
            #nn.Linear(256,200),
            #nn.Sigmoid(),
            nn.Linear(256,129)
            
            )
            

    def forward(self, x):
        x = self.encoder(x)
        #print(x)
        x = self.last(x)
        return x