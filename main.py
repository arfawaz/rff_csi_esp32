#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 23:40:19 2025

@author: fawaz
"""
from csi_dataset_creator_fixed_id import process_csv_fixed_id
from csi_dataset_creator_fixed_id_uniform_sampling import process_csv_fixed_id_uniform_sampling
from csi_dataset_creator import process_csv
from mean_norm import mean_norm
from train_test import train, test
from train_test_loader import train_test_loader
from models import SimpleCNN
import torch.nn as nn
import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#%% Training and Testing 

# Prompt the user for the file path
file_path = input("Please enter the file path to the CSV file: ")

# Process the CSV file
data, labels = process_csv_fixed_id_uniform_sampling(file_path = file_path , mac_id_list = \
["00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"00:FC:BA:38:4B:02", \
"2A:C8:A7:E1:8F:F0"], max_samples_per_mac=1000)
data = data.unsqueeze(1)
dataset = mean_norm(data)
train_loader, test_loader = train_test_loader(dataset, labels)

num_classes = 5
learning_rate = 0.001
num_epochs = 50

# Model setup
model = SimpleCNN(num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Model Training
model.train()
train(model=model, train_loader=train_loader, test_loader=test_loader, criterion=criterion, optimizer=optimizer, num_epochs=num_epochs)

# Model Testing
model.eval()
_ = test(model, test_loader)


#%% Testing on new test dataset

file_path = input("Please enter the file path to the test CSV file: ")
data, labels = process_csv(file_path = file_path , mac_id_list = ["34:5F:45:A8:3C:19", \
"8C:4F:00:3C:BF:4D", \
"3C:8A:1F:90:E3:31", \
"34:5F:45:A9:A4:19", \
"20:43:A8:64:3A:C1"], max_samples_per_mac=5000)
data = data.unsqueeze(1)
dataset = mean_norm(data)
train_loader, test_loader = train_test_loader(dataset, labels)

model.eval()
_ = test(model, train_loader)












