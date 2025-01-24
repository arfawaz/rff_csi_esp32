#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:30:56 2025

@author: fawaz
"""

###############################################################################
# imports
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


###############################################################################


#1) train_test_loader
"""
Takes in data and corresponding labels and return a train_loader and test_loader
that can be used on deep learning models.
-------------------------------------------------------------------------------

Parameters:
- data (torch.Tensor)
- labels (torch.Tensor)
- train_percent (float)
- batch_size (int)

Returns:
- train_loader ()

"""

def train_test_loader(data,labels,train_percent = 0.8,batch_size=16):
    dataset = TensorDataset(data, labels) #Converting to Tensors
    
    # Split into train and test sets (80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    #Create DataLoader objects from tain_dataset and test_dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader