# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 16:28:57 2025

@author: fawaz243
"""

import torch

@torch.no_grad()
def evaluate_zero_shot(model,loader, device):
    """
    Classification accuracy by cosine sim: z_csi vs all class embeddings.
    """
    
    model.eval()
    C = model.class_table().size(0)
    E = model.class_table().to(device) # The trained label encoder matrix of size (C,d) where C is number of classes and d is the feature dimension.
    corr, total = 0, 0
    
    for xb, yb in loader:
        xb = xb.to(device) 
        yb = yb.to(device)
        z = model.encode_csi(xb) # output of the encoder after passing csi data in xb. The output is of size (B,d) where B is the batch size and d is the feature dimension.
        logits = z @ E.t()
        pred = logits.argmax(dim = -1)
        corr += (pred==yb).sum().item()
        total += yb.numel()
    return corr/max(total,1)