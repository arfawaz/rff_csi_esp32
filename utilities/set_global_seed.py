# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 14:30:53 2025

@author: fawaz243
"""

def set_global_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False