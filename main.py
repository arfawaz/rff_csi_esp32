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
from models import SimpleCNN, vit_model_2, ResNet50CSI, CSIEncoder, LabelEmbedder, CSIResNet50Encoder, CSI_CLIP
from torch.utils.data import Subset, DataLoader, TensorDataset, random_split
from transformers import AdamW
import torch.nn as nn
import torch
import torch.optim as optim
from losses import clip_loss
from clip_dataset import CSIDataset
from evaluate_zero_shot import evaluate_zero_shot
from train_test import train_clip
from stratified_train_val_indices import stratified_train_val_indices
from make_loaders_for_dataset import make_loaders_for_dataset
import os, random
from collections import defaultdict
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights
from set_global_seed import set_global_seed
from losses import clip_loss
from evaluate_zero_shot import evaluate_zero_shot

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

#%% Training and testing the CSI_CLIP based model, simpleCNN, ResNet50CSI on the same dataset.

# This cell is used for leaoding the same dataset for cnn, ResNet50CSI and clip model.

SEED = 20250910
SHARED_SPLIT_FILE = f"shared_train_val_split_seed{SEED}.pt"

set_global_seed(SEED)


file_path = input("Please enter the file path to CSV files for train data: ")

MAC_ID_LIST = [
    "00:FC:BA:38:4B:00",
    "00:FC:BA:38:4B:01",
    "00:FC:BA:38:4B:02",
    "6C:B2:AE:39:1A:A0",
    "6C:B2:AE:39:1A:A1",
    "70:0F:6A:DE:EC:A0",
    "70:0F:6A:DE:EC:A1",
    "70:0F:6A:DE:EC:A2",
]

data, labels = process_csv_fixed_id_uniform_sampling_rssi(file_path = file_path , mac_id_list = MAC_ID_LIST, max_samples_per_mac=100000000)
    
print("Done data loading Data and Labels from CSV")



# --------------------------
# Build full TRAIN sets (once)
#   data:   [N, 64, 2]
#   labels: [N]
# --------------------------
labels = labels.long()
N = data.size(0)

# CNN view: [N,1,64,2] + mean_norm()
x_cnn_full = mean_norm(data.unsqueeze(1).float())
cnn_full_ds = TensorDataset(x_cnn_full, labels)

# CLIP base view: [N,2,64] (normalize/augment inside wrapper)
x_clip_base = data.permute(0, 2, 1).float()  # [N,2,64]

if os.path.exists(SHARED_SPLIT_FILE):
    split = torch.load(SHARED_SPLIT_FILE)
    train_idx, val_idx = split["train_idx"], split["val_idx"]
    assert max(train_idx + val_idx) < N, \
        "Saved split indices exceed current TRAIN dataset size. Delete split file and re-run."
else:
    train_idx, val_idx = stratified_train_val_indices(labels, train_ratio=0.9, seed=SEED)
    torch.save({"train_idx": train_idx, "val_idx": val_idx}, SHARED_SPLIT_FILE)

print(f"[TRAIN CSV] TRAIN={len(train_idx)} | VAL={len(val_idx)}  (seed={SEED})")

# CNN loaders (use these in your CNN train/test functions)
cnn_train_loader, cnn_val_loader = make_loaders_for_dataset(cnn_full_ds, train_idx, val_idx, batch_size=64)

# CLIP datasets (train uses aug; val no aug)
clip_train_full = CSIDataset(x_clip_base, labels, normalize=True, augment=True);  clip_train_full.train_mode = True
clip_val_full   = CSIDataset(x_clip_base, labels, normalize=True, augment=False); clip_val_full.train_mode   = False
clip_train_ds = Subset(clip_train_full, train_idx)
clip_val_ds   = Subset(clip_val_full,   val_idx)

print("[TRAIN/VAL] Shared splits ready for CNN and CLIP.")

# --------------------------
# TEST set: load from a NEW CSV ONE TIME and reuse for BOTH
# --------------------------
# Ensure deterministic sub-sampling inside your CSV loader
random.seed(SEED)

test_file_path = input("Please enter the file path to the TEST CSV file: ")

test_data, test_labels = process_csv_fixed_id_uniform_sampling_rssi(
    file_path=test_file_path,
    mac_id_list=MAC_ID_LIST,
    max_samples_per_mac=1_000_000
)
assert test_data is not None and test_labels is not None, "No valid test data found in TEST CSV."
test_labels = test_labels.long()


# CNN TEST loader (entire TEST set)
x_cnn_test = mean_norm(test_data.unsqueeze(1).float())
cnn_test_ds = TensorDataset(x_cnn_test, test_labels)
cnn_test_loader = DataLoader(cnn_test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

# CLIP TEST loader (entire TEST set; normalized, no aug)
x_clip_test = test_data.permute(0, 2, 1).float()  # [M,2,64]
clip_test_ds = CSIDataset(x_clip_test, test_labels, normalize=True, augment=False)
clip_test_ds.train_mode = False
clip_test_loader = DataLoader(clip_test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

print(f"[TEST CSV] TEST={len(test_labels)} (shared for both models)")

#%% Training the cnn-3 model with training data loaded from previous cell. Testing the cnn-3 model with testing data loaded from previous cell. 

print("Starting training the loaded data on cnn-3 model")
num_classes = 8
learning_rate = 0.001
num_epochs = 2
# Model setup
model_1 = SimpleCNN(num_classes)
model_1 = model_1.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_1.parameters(), lr=learning_rate)

# Model Training
model_1.train()
train(model=model_1, 
      train_loader=cnn_train_loader, 
      test_loader=cnn_val_loader, 
      criterion=criterion, 
      optimizer=optimizer, 
      num_epochs=num_epochs)

print("Trianing finished on cnn-3 model")

print("Testing the data cnn-3 model on the loaded testing data")
# Model Testing
model_1.eval()
_ = test(model_1, cnn_test_loader)

print("Testing finished on cnn-3 model")

#%% Training the ResNet50CSI model with training data loaded from previous cell. Testing the cnn-3 model with testing data loaded from previous cell.

model_3 = ResNet50CSI(num_classes=num_classes, pretrained=True).to(device)
resnet_lr = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_3.parameters(), lr=resnet_lr)

# Train
model_3.train()
train(
    model=model_3,
    train_loader=cnn_train_loader,
    test_loader=cnn_val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=1
)

# Evaluate on primary test set
model_3.eval()
_ = test(model_3, cnn_test_loader)


#%% Training the CSI_CLIP model with cnn-3 CSI encoder training data loaded from previous cell. Testing the CSI_CLIP model with cnn-3 CSI encoder model with testing data loaded from previous cell.

# Assuming you have CSI_CLIP, CSIEncoder, LabelEmbedder, evaluate_zero_shot, and train(...) from earlier
num_classes = len(MAC_ID_LIST)
clip_model = CSI_CLIP(
    csi_encoder=CSIEncoder(in_ch=2, proj_dim=256),
    label_encoder=LabelEmbedder(num_classes=num_classes, dim=256),
)

trained_clip = train_clip(
    clip_model, clip_train_ds, clip_val_ds,
    epochs=1, batch_size=64, lr=1e-3, wd=1e-4, num_workers=0, device="cuda"
)

# Final test on the external TEST CSV (same samples as CNN)
clip_test_acc = evaluate_zero_shot(trained_clip, clip_test_loader, device=("cuda" if torch.cuda.is_available() else "cpu"))
print(f"[CLIP] TEST top-1: {clip_test_acc*100:.2f}%")


#%% Training the CSI_CLIP model with  CSIResNet50 encoder training data loaded from previous cell. Testing CSI_CLIP model with  CSIResNet50 encoder model with testing data loaded from previous cell. 

num_classes = len(MAC_ID_LIST)

clip_model_resnet = CSI_CLIP(
    csi_encoder=CSIResNet50Encoder(proj_dim=256, pretrained=True, freeze_backbone_bn=False),
    label_encoder=LabelEmbedder(num_classes=num_classes, dim=256),
).to(device)

trained_clip = train_clip(
    clip_model_resnet, clip_train_ds, clip_val_ds,
    epochs=1, batch_size=64, lr=1e-4, wd=1e-4, num_workers=0, device="cuda"
)

# Final test on the external TEST CSV (same samples as CNN)
clip_test_acc = evaluate_zero_shot(trained_clip, clip_test_loader, device=("cuda" if torch.cuda.is_available() else "cpu"))
print(f"[CLIP] TEST top-1: {clip_test_acc*100:.2f}%")   