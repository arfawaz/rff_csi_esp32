#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:41:29 2025

@author: fawaz
"""

###############################################################################
# imports
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


###############################################################################

# 1) train

"""
This function is used to train the model passed into it.
-------------------------------------------------------------------------------

Parameters:
- model (torch.nn.module)
- train_loader (torch.utils.data.DataLoader)
- test_loader (torch.utils.data.DataLoader)
- criterion (nn.function)
- optimizer (torch.optim)
- num_epochs (int)

Returns:
- Nothing is returned

"""

def train(model, train_loader, test_loader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            #inputs = inputs.unsqueeze(0)
            #print(f"shape of input is: {inputs.shape}")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {test(model,test_loader)}")

###############################################################################

# 2) test

"""
This function is used to test the data on model and test data passed into it.
-------------------------------------------------------------------------------

Parameters:
- model (torch.nn.module)
- loader (torch.utils.data.DataLoader)

Returns:
- accuracy (float)

"""


def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
           
            #print(f"shape of input is: {inputs.shape}")
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy