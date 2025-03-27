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
from train_vit_model_2 import train_vit_model_2
from test_vit_model_2 import test_vit_model_2
from CustomDataset_vit_model_2 import CustomDataset_vit_model_2
from models import SimpleCNN, vit_model_2
from torch.utils.data import Subset, DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#%% Loading data and labels for CNN and vit_model_2

# Prompt the user for the file path
file_path = input("Please enter the file path to the CSV file: ")

# Process the CSV file
data, labels = process_csv_fixed_id_uniform_sampling(file_path = file_path , mac_id_list = \
["00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"00:FC:BA:38:4B:02", \
"70:DB:98:9E:3A:A0", \
"70:DB:98:9E:3A:A1"], max_samples_per_mac=1000)
    
print("Done data loading")
    
###############################################################################
    
#%% Training and testing on SimpleCNN
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


#%% Training and testing on vit_model_2
dataset_vit_model_2 = CustomDataset_vit_model_2(data, labels)
batch_size = 16
# Split into train and test datasets
train_size = int(0.9 * len(dataset_vit_model_2))
test_size = len(dataset_vit_model_2) - train_size
train_dataset_vit_model_2, test_dataset_vit_model_2 = random_split(dataset_vit_model_2, [train_size, test_size])
train_loader_vit_model_2 = DataLoader(train_dataset_vit_model_2, batch_size=batch_size, shuffle=True)
test_loader_vit_model_2 = DataLoader(test_dataset_vit_model_2, batch_size=batch_size, shuffle=False)

num_classes = 5
learning_rate = 5e-5
num_epochs = 50

model = vit_model_2(num_classes = num_classes)
model.to(device)

# Set up the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()


# Training the model
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Training
    train_loss, train_accuracy = train_vit_model_2(model = model, train_loader = train_loader_vit_model_2, optimizer = optimizer, loss_fn = loss_fn , device = device)
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    # Testing
    test_loss, test_accuracy = test_vit_model_2(model = model, test_loader = test_loader_vit_model_2, loss_fn = loss_fn, device = device)
    print(f"Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_accuracy:.2f}%")


#%% Testing on new test dataset

file_path = input("Please enter the file path to the test CSV file: ")
data, labels =  process_csv_fixed_id_uniform_sampling(file_path = file_path , mac_id_list = ["34:5F:45:A8:3C:19", \
"8C:4F:00:3C:BF:4D", \
"3C:8A:1F:90:E3:31", \
"34:5F:45:A9:A4:19", \
"20:43:A8:64:3A:C1"], max_samples_per_mac=5000)
data = data.unsqueeze(1)
dataset = mean_norm(data)
train_loader, test_loader = train_test_loader(dataset, labels)

model.eval()
_ = test(model, train_loader)












