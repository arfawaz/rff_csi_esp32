#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 00:26:08 2025

@author: fawaz
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
#%% CSIEncoder
class CSIEncoder(nn.Module):
    
    """
   1D Conv encoder for CSI shaped [B, 2, 64] (channels = [magnitude, angle]).
   You can swap this to your complex CNN or a Transformer later.
   """
    
    def __init__(self,in_ch=2, proj_dim = 256):
        super().__init__()
        
        self.feat = nn.Sequential(
            nn.Conv1d(in_channels = in_ch, out_channels, = 64, kernel_size = 5, padding = 2), nn.ReLU(),
            nn.Conv1d(in_channels = 64, out_channels=128, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # -> [B, 128, 1]
        )
        
        self.proj = nn.Linear(128,proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        
    
    def forward(self,x): # x: [B, 2, 64]
        h = self.feat(x).squeeze(-1) # [B, 128]
        z = self.norm(self.proj(h)) # [B, d]
        return F.normalize(z,dim = -1) # unit-norm embeddings
    

#%%

class LabelEncoder(nn.Module):
     """
    Learnable embedding per MAC/class. Works great when the class set is fixed.
    """
    
    def __init__(self,num_classes, dim = 256):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=num_classes, embedding_dim=dim)
        nn.init.normal_(self.emb.weight, std=0.02)
        
    def forward(self,y):
        z = self.emb(y)
        return F.normalize(z, dim=-1)
    
    
    def table(self):
        
        return F.normalize(self.emb.weight,dim=1)
    
#%%
    
class CSI_CLIP(nn.Module):
    def __init__(self,csi_encoder: nn.Module, label_encoder: nn.Module):
        super().__init__()
        self.csi = csi_encoder
        self.txt = label_encoder
        self.logit_scale = nn.Parameter()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))  # ~ ln(14.285)
    
    def forward(self,csi_batch, y_batch):
        
        """
        Returns CLIP logits [B, B] comparing CSI ↔ labels of the SAME batch.
        """
        
        zc = self.csi(csi_batch) # [B, d]
        zt = self.txt(y_batch) # [B, d]
        scale - self.logit_scale.exp().clamp(max=100.0)
        return scale * zc @ zt.t()  # [B, B]
    
    @torch.no_grad()
    def encode_csi(self, csi_batch):
        return self.csi(csi_batch)
    
    @torch.no_grad()
    def class_table(self):
        return self.txt.table()

#%%
def clip_loss(logits):
    
    B = logits.size(0)
    target = torch.arange(B, device = logits.device)
    loss_i = F.cross_entropy(logits, target)
    loss_t = F.cross_entropy(logits.t(), target)
    return 0.5 * (loss_i + loss_t)
    
    
#%%
    
class CDIDataset(Dataset):
    
    def __init__(self, data, labels, normalize = True, augment = False):
        super().__init__()
        assert data.ndim ==3 and data.shape[1:] == (64,2), \
            f"Expected [N,64,2], got {tuple(data.shape)}"
        
        self.x = data.permute(0,2,1).float() # [N,64,2] -> [N,2,64]
        self.y = labels.long()
        self.normalize = normalize
        self.augment = augment
    
    def __len__(self):
        return self.x.size(0)
    
    def _normalize(self,x):
        # x: [2, 64]  -> per-channel z-score across subcarriers
        mean = x.mean(dim =1, keepdim = True)
        std = x.std(dim = -1, keepdim = True).clamp_mim(1e-6)
        return (x-mean)/std
    
    def _augment(self, x):
        # light noise; you can add phase jitter or subcarrier dropout later
        if torch.rand(1).item() < 0.5:
            x = x + 0.01 * torch.randn_like(x)
        return x
    
    def __getitem__(self,idx):
        xi, yi = self.x[idx], self.y[idx]
        if self.normalize:
            xi = self._normalize(xi)
        if self.augment:
            xi = self.augment(xi)
            
        return xi, yi
    
    # toggle for train/eval augment policy
    @property
    def train_mode(self): return getattr(self, "_train_mode", False)
    @train_mode.setter
    def train_mode(self, v): self._train_mode = bool(v)

        
    
    
    
    
    
    
    
    
    