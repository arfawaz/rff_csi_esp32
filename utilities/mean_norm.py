#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:46:53 2025

@author: fawaz
"""
###############################################################################
# imports
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


###############################################################################
###############################################################################

# 5) mean_norm

"""
This function is used to mean normalize the data passed into it.
-------------------------------------------------------------------------------

Parameters:
- data (torch.tensor)

Returns:
- data (torch.tensor)


"""

def mean_norm(data):
    mean = data.mean()
    std = data.std()
    data = (data - mean) / std
    return data

###############################################################################
