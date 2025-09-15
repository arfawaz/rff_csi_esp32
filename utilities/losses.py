# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 15:47:17 2025

@author: fawaz243
"""
import torch
import torch.nn.functional as F
# clip loss
def clip_loss(logits):
    
    B = logits.size(0)
    target = torch.arange(B, device = logits.device)
    loss_i = F.cross_entropy(logits, target)
    loss_t = F.cross_entropy(logits.t(), target)
    return 0.5 * (loss_i + loss_t)