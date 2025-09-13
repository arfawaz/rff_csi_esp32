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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            inputs, labels = inputs.to(device), labels.to(device)
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
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

###############################################################################

# 3) train_clip

def train_clip(model, train_ds, val_ds, *, epochs = 10, batch_size = 64, lr = 1e-3, wd = 1e4, num_workers = 0, device = 'cuda'):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # mark augment mode on train set only
    train_ds.train_mode = False
    val_ds.train_mode = False
    
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True,
                              drop_last = True, num_workers = num_workers, pin_memory = True)
    
    val_loader = DataLoader(val_ds, batch_size = batch_size, shuffle = False,
                              drop_last = False, num_workers = num_workers, pin_memory = True)
    
    opt = torch.optim.AdamW([
        {"params":model.csi.parameters()},
        {"params":model.txt.parameters()},
        {"params":[model.logit_scale]},
        ], lr = lr, weight_decay = wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    best_val = 0.0
    
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking = True)
            yb = yb.to(device, non_blocking = True)
            
            logits = model(xb, yb)
            loss = clip_loss(logits)
            
            opt.zero_grad(set_to_none = True)
            loss.backward()
            opt.step()
            total_loss +=loss.item() * xb.size(0)
            
        scheduler.step()
        avg_loss = total_loss / (len(train_loader.dataset) // train_loader.batch_size * train_loader.batch_size)
        
        val_acc = evaluate_zero_shot(model, val_loader, device)
        
        print(f"[Epoch {ep:02d}] loss={avg_loss:.4f}  val@1={val_acc*100:.2f}%  logit_scale={model.logit_scale.exp().item():.3f}")
        
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict()}, "best_csi_clip.pth")
        
    print(f"Best val@1: {best_val*100:.2f}%")
    return model
