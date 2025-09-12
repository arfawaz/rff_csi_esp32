#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 23:40:19 2025

@author: fawaz
"""
from csi_dataset_creator_fixed_id import process_csv_fixed_id
from csi_dataset_creator_fixed_id_uniform_sampling import process_csv_fixed_id_uniform_sampling
from process_csv_fixed_id_uniform_smapling_rssi import process_csv_fixed_id_uniform_sampling_rssi
from csi_dataset_creator import process_csv
from mean_norm import mean_norm
from train_test import train, test
from train_test_loader import train_test_loader
from train_vit_model_2 import train_vit_model_2
from test_vit_model_2 import test_vit_model_2
from CustomDataset_vit_model_2 import CustomDataset_vit_model_2
from models import SimpleCNN, vit_model_2, ResNet50CSI
from torch.utils.data import Subset, DataLoader, TensorDataset, random_split
from transformers import AdamW
import torch.nn as nn
import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################################################################

#%% Loading data and labels for CNN and vit_model_2

# Prompt the user for the file path
file_path = input("Please enter the file path to the CSV file: ")

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\train

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\transmitter_number_test\train/train.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\train\all_positions_sampled_merged.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\train

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\train\train.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p1_p2_p3_p4_p5_p6\train\all_positions_sampled_merged.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\all_poistions\train\all_positions_sampled_merged.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p7\train\sampled_p7.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p7_p8_p9_p10\train\sampled_p7_p8_p9_10_merged.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\range_test\p1_p2\train.csv

# Process the CSV file
data, labels = process_csv_fixed_id_uniform_sampling_rssi(file_path = file_path , mac_id_list = [
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"00:FC:BA:38:4B:02", \
"6C:B2:AE:39:1A:A0", \
"6C:B2:AE:39:1A:A1", \
"70:0F:6A:DE:EC:A0", \
"70:0F:6A:DE:EC:A1", \
"70:0F:6A:DE:EC:A2", \
#"00:FC:BA:27:63:00", \
#"00:FC:BA:27:63:01", \
#"00:FC:BA:27:63:02"
], max_samples_per_mac=100000000)
    
print("Done data loading")
    
###############################################################################
    
#%% Training and testing on SimpleCNN
data_simplecnn = data.unsqueeze(1).clone()
dataset = mean_norm(data_simplecnn)
train_loader, test_loader = train_test_loader(dataset, labels, train_percent = 0.9)

num_classes = 8
learning_rate = 0.001
num_epochs = 10

# Model setup
model_1 = SimpleCNN(num_classes)
model_1 = model_1.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_1.parameters(), lr=learning_rate)

# Model Training
model_1.train()
train(model=model_1, train_loader=train_loader, test_loader=test_loader, criterion=criterion, optimizer=optimizer, num_epochs=num_epochs)

# Model Testing
model_1.eval()
_ = test(model_1, test_loader)

###############################################################################

#%% Training and testing on vit_model_2
data_vit_model_2 = data.clone()
dataset_vit_model_2 = CustomDataset_vit_model_2(data_vit_model_2, labels)
batch_size = 64
# Split into train and test datasets
train_size = int(0.9 * len(dataset_vit_model_2))
test_size = len(dataset_vit_model_2) - train_size
train_dataset_vit_model_2, test_dataset_vit_model_2 = random_split(dataset_vit_model_2, [train_size, test_size])
train_loader_vit_model_2 = DataLoader(train_dataset_vit_model_2, batch_size=batch_size, shuffle=True)
test_loader_vit_model_2 = DataLoader(test_dataset_vit_model_2, batch_size=batch_size, shuffle=False)

num_classes = 12
learning_rate = 5e-5
num_epochs = 10

model_2 = vit_model_2(num_classes = num_classes)
model_2.to(device)

# Set up the optimizer and loss function
optimizer = AdamW(model_2.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()


# Training the model
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Training
    train_loss, train_accuracy = train_vit_model_2(model = model_2, train_loader = train_loader_vit_model_2, optimizer = optimizer, loss_fn = loss_fn , device = device)
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    # Testing
    test_loss, test_accuracy = test_vit_model_2(model = model_2, test_loader = test_loader_vit_model_2, loss_fn = loss_fn, device = device)
    print(f"Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_accuracy:.2f}%")

###############################################################################

#%% Testing SimpleCNN on new test dataset 

file_path = input("Please enter the file path to the test CSV file: ")

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\test
#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\transmitter_number_test\test\test.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\test\test.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p1_p2_p3_p4_p5_p6_p7\test\all_positions_sampled_merged.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_test_set\systematic_test_merged\systematic_test_merged.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p7\test\sampled_p7.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\all_poistions\test\all_positions_sampled_merged.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p1\nearby_positions\

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\test\test_sampled\test.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\range_test\test.csv

data_cnn_test, labels_cnn_test =  process_csv_fixed_id_uniform_sampling_rssi(file_path = file_path , mac_id_list = [
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"00:FC:BA:38:4B:02", \
"6C:B2:AE:39:1A:A0", \
"6C:B2:AE:39:1A:A1", \
"70:0F:6A:DE:EC:A0", \
"70:0F:6A:DE:EC:A1", \
"70:0F:6A:DE:EC:A2", \
#"00:FC:BA:27:63:00", \
#"00:FC:BA:27:63:01", \
#"00:FC:BA:27:63:02"
], max_samples_per_mac=1000000)
data_cnn_test = data_cnn_test.unsqueeze(1)
dataset_cnn_test = mean_norm(data_cnn_test)
train_loader_cnn_test, test_loader_cnn_test = train_test_loader(dataset_cnn_test, labels_cnn_test)


model_1.eval()
_ = test(model_1, train_loader_cnn_test)

#C:/Users/fawaz/OneDrive - University of South Florida/Desktop/USF/SEMESTER 1 - FALL 23/DIRECTED RESEARCH/projects_on_git/rff_csi_esp32/csi_data_collected/systematic_collection/27_mar_25_p2_12_30_03_30.csv

###############################################################################

#%% Testing vit_model_2 on new test dataset

file_path = input("Please enter the file path to the test CSV file: ")

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_test_set\systematic_test_merged\systematic_test_merged.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\test\all_positions_sampled_merged.csv

data_vit_model_2_testing, labels_vit_model_2_testing =  process_csv_fixed_id_uniform_sampling_rssi(file_path = file_path , mac_id_list = [ \
    "6C:B2:AE:39:1A:A0", 
    "70:0F:6A:DE:EC:A0", 
    "70:0F:6A:DE:EC:A1", 
    "6C:B2:AE:39:1A:A1", 
    "70:0F:6A:DE:EC:A2", 
    "6C:B2:AE:39:1A:A2", 
    #"C8:28:E5:44:3B:00", 
    "00:FC:BA:38:4B:00", 
    "00:FC:BA:38:4B:01", 
    "70:0F:6A:FC:51:80", 
    "00:FC:BA:38:4B:02", 
    "84:3D:C6:5F:5D:50" 
], max_samples_per_mac=100000)
    
    
dataset_vit_model_2_testing = CustomDataset_vit_model_2(data_vit_model_2_testing, labels_vit_model_2_testing)
batch_size = 16
# Split into train and test datasets
train_size_vit_model_2_testing = int(0.1* len(dataset_vit_model_2_testing))
test_size_vit_model_2_testing = len(dataset_vit_model_2_testing) - train_size_vit_model_2_testing
train_dataset_vit_model_2_testing, test_dataset_vit_model_2_testing = random_split(dataset_vit_model_2_testing, [train_size_vit_model_2_testing, test_size_vit_model_2_testing])
train_loader_vit_model_2_testing = DataLoader(train_dataset_vit_model_2_testing, batch_size=batch_size, shuffle=True)
test_loader_vit_model_2_testing = DataLoader(test_dataset_vit_model_2_testing, batch_size=batch_size, shuffle=False)

test_loss_vit_model_2_testing, test_accuracy_vit_model_2_testing = test_vit_model_2(model = model_2, test_loader = train_loader_vit_model_2_testing, loss_fn = loss_fn, device = device)
print(f"Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_accuracy_vit_model_2_testing:.2f}%")

#C:/Users/fawaz/OneDrive - University of South Florida/Desktop/USF/SEMESTER 1 - FALL 23/DIRECTED RESEARCH/projects_on_git/rff_csi_esp32/csi_data_collected/systematic_collection/27_mar_25_p2_12_30_03_30.csv

###############################################################################



#%% Training and testing on ResNet50 (CSI adapter)
# Uses the SAME loaders as SimpleCNN:
#   train_loader: batches of x shaped [B, 1, 64, 2], y shaped [B]
#   test_loader:  same shapes as above

# NOTE on shapes through the adapter:
#   In-batch x: [B, 1, 64, 2]
#     ⇢ resize bilinear → [B, 1, 224, 224]
#     ⇢ repeat channels → [B, 3, 224, 224]
#     ⇢ ResNet-50 → logits [B, num_classes]
data_simplecnn = data.unsqueeze(1).clone()
dataset = mean_norm(data_simplecnn)
train_loader, test_loader = train_test_loader(dataset, labels, train_percent = 0.9)
num_classes = 8
resnet_lr = 1e-4
resnet_epochs = 4  # adjust as needed

model_3 = ResNet50CSI(num_classes=num_classes, pretrained=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_3.parameters(), lr=resnet_lr)

# Train
model_3.train()
train(
    model=model_3,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=resnet_epochs
)

# Evaluate on primary test set
model_3.eval()
_ = test(model_3, test_loader)

# (Optional) If you also have a secondary loader like `train_loader_cnn_test`
# from your SimpleCNN section, evaluate there too.
try:
    model_3.eval()
    _ = test(model_3, train_loader_cnn_test)
except NameError:
    pass
###############################################################################


#%% Testing RESNET50 on new test dataset 

file_path = input("Please enter the file path to the test CSV file: ")

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\test\test.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\transmitter_number_test\test\test.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\test\test.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p1_p2_p3_p4_p5_p6_p7\test\all_positions_sampled_merged.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\systematic_collection_test_set\systematic_test_merged\systematic_test_merged.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p7\test\sampled_p7.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\all_poistions\test\all_positions_sampled_merged.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p1\nearby_positions\

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\test\test_sampled\test.csv

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\range_test\test.csv

data_cnn_test, labels_cnn_test =  process_csv_fixed_id_uniform_sampling_rssi(file_path = file_path , mac_id_list = [
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"00:FC:BA:38:4B:02", \
"6C:B2:AE:39:1A:A0", \
"6C:B2:AE:39:1A:A1", \
"70:0F:6A:DE:EC:A0", \
"70:0F:6A:DE:EC:A1", \
"70:0F:6A:DE:EC:A2", \
#"00:FC:BA:27:63:00", \
#"00:FC:BA:27:63:01", \
#"00:FC:BA:27:63:02"
], max_samples_per_mac=1000000)
data_cnn_test = data_cnn_test.unsqueeze(1)
dataset_cnn_test = mean_norm(data_cnn_test)
train_loader_cnn_test, test_loader_cnn_test = train_test_loader(dataset_cnn_test, labels_cnn_test)


model_3.eval()
_ = test(model_3, train_loader_cnn_test)

#C:/Users/fawaz/OneDrive - University of South Florida/Desktop/USF/SEMESTER 1 - FALL 23/DIRECTED RESEARCH/projects_on_git/rff_csi_esp32/csi_data_collected/systematic_collection/27_mar_25_p2_12_30_03_30.csv

###############################################################################

