# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 16:02:55 2025

@author: fawaz243
"""

# CLIP normalization/augmentation wrapper
class CSIDataset(torch.utils.data.Dataset):
    def __init__(self, x_2x64, y, normalize=True, augment=False):
        self.x = x_2x64
        self.y = y.long()
        self.normalize = normalize
        self.augment = augment
        self._train_mode = False
    def __len__(self): return self.x.size(0)
    def _normalize(self, x):
        m, s = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - m) / s
    def _augment(self, x):
        if torch.rand(1).item() < 0.5:
            x = x + 0.01 * torch.randn_like(x)
        return x
    @property
    def train_mode(self): return self._train_mode
    @train_mode.setter
    def train_mode(self, v): self._train_mode = bool(v)
    def __getitem__(self, i):
        xi, yi = self.x[i], self.y[i]
        if self.normalize: xi = self._normalize(xi)
        if self.augment and self._train_mode: xi = self._augment(xi)
        return xi, yi
