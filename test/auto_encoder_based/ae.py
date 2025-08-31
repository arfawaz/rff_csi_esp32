#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 13:44:43 2025

@author: fawaz
"""

import torch
import torch.nn as nn
class CSIAutoEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.enc_conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1) # input(B,2,64) -- >output(B,16,64)
        self.enc_conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1) # input(B,16,64) --> output(B,32,64)
        self.enc_flat = nn.Flatten() # input(B,32,64) -->  output(B,32*64)
        self.enc_fc = nn.Linear(32*64,latent_dim) # input(B,32*64) -->  output(B,latent_dim)
        
        self.dec_fc = nn.Linear(latent_dim,32*64) # input(B,latent_dim) --> output(B,32*64)
        self.dec_unflat = nn.Unflatten(1, (32,64)) # input(B,32*64) --> ouptut(B,32,64)
        self.dec_conv1 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding = 1) #input(B,32,64) --> output(B,16,64)
        self.dec_conv2 = nn.Conv1d(in_channels=16, out_channels=2, kernel_size=3, padding = 1) #input(B,16,64) --> output(B,2,64)
        
        
    def encode(self,x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        z = self.enc_fc(self.enc_flat(x))