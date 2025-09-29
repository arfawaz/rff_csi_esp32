#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 23:40:19 2025

@author: fawaz
"""
# --- make project_root importable everywhere (terminal/editor/new machine) ---
# ---- make project_root importable (works in Spyder/terminal/any CWD) ----
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent  # folder that contains /utilities and /models
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# -------------------------------------------------------------------------

from utilities.csi_dataset_creator_fixed_id import process_csv_fixed_id
from utilities.csi_dataset_creator_fixed_id_uniform_sampling import process_csv_fixed_id_uniform_sampling
from utilities.process_csv_fixed_id_uniform_smapling_rssi import process_csv_fixed_id_uniform_sampling_rssi
from utilities.csi_dataset_creator import process_csv
from utilities.mean_norm import mean_norm
from utilities.train_test import train, test
from utilities.train_test_loader import train_test_loader
from utilities.train_vit_model_2 import train_vit_model_2
from utilities.test_vit_model_2 import test_vit_model_2
from utilities.CustomDataset_vit_model_2 import CustomDataset_vit_model_2
from models.models import SimpleCNN, vit_model_2, ResNet50CSI, CSIEncoder, LabelEmbedder, CSIResNet50Encoder, CSI_CLIP, CNN_2, LabelHexProjector, LabelHexPlusLoc
from torch.utils.data import Subset, DataLoader, TensorDataset, random_split
from transformers import AdamW
import torch.nn as nn
import torch
import torch.optim as optim
from utilities.losses import clip_loss
from utilities.clip_dataset import CSIDataset
from utilities.evaluate_zero_shot import evaluate_zero_shot
from utilities.train_test import train_clip
from utilities.stratified_train_val_indices import stratified_train_val_indices
from utilities.make_loaders_for_dataset import make_loaders_for_dataset
import os, random
from collections import defaultdict
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights
from utilities.set_global_seed import set_global_seed
from utilities.losses import clip_loss
from utilities.evaluate_zero_shot import evaluate_zero_shot
import json, datetime
from pathlib import Path
import optuna
from optuna.pruners import MedianPruner
from torch.utils.data import DataLoader

# from utilities.optuna_helper_functions import (
#     _jsonify,
#     save_study_best,
#     save_checkpoint,
#     HAS_DATASETS,
#     make_cls_loaders_for_trial,
#     objective_cnn2_classifier,
#     run_study_cnn2,
#     retrain_and_test_cnn2,
#     objective_resnet50csi_classifier,
#     run_study_resnet50csi,
#     retrain_and_test_resnet50csi,
#     train_clip_once,
#     suggest_common_hparams,
#     build_cnn_encoder,
#     build_resnet_encoder,
#     objective_clip_cnn_mac,
#     objective_clip_cnn_mac_loc,
#     objective_clip_resnet_mac,
#     objective_clip_resnet_mac_loc,
#     run_study,
#     retrain_and_test,
# )
from utilities.optuna_helper_functions import (
    # persistence + runner
    save_study_best, run_study,

    # CNN_2
    make_objective_cnn2_classifier, retrain_and_test_cnn2,

    # CLIP objectives (factories) + final eval
    make_objective_clip_cnn_mac,
    make_objective_clip_cnn_mac_loc,
    make_objective_clip_resnet_mac,
    make_objective_clip_resnet_mac_loc,
    retrain_and_test,

    # ResNet50CSI classifier
    make_objective_resnet50csi_classifier,
    retrain_and_test_resnet50csi,
)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################################################################

#%% Loading data and labels for CNN and vit_model_2

if False:
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
if False:
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
if False:
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

if False:    
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

if False:
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

if False: 
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

if False:
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

if False:
    # This cell is used for leaoding the same dataset for cnn, ResNet50CSI and clip model.
    
    SEED = 20250910
    SHARED_SPLIT_FILE = f"shared_train_val_split_seed{SEED}.pt"
    
    set_global_seed(SEED)
    
    file_path = r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\openai_clip_based\individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\train\train.csv"
    #file_path = input("Please enter the file path to CSV files for train data: ")
    
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
    
    test_file_path = r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\openai_clip_based\individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\test\test.csv"
    #test_file_path = input("Please enter the file path to the TEST CSV file: ")
    
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


#%% Training and testing the CSI_CLIP based model, simpleCNN, ResNet50CSI on the same dataset.
# Uses explicit TRAIN / VAL / TEST CSVs (no split). All models see the same samples.

if True:
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = Path(os.getenv("RFF_CSI_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
    SEED = 20250910
    set_global_seed(SEED)
    
    # --------------------------
    # File paths
    # --------------------------
    train_file_path = DATA_DIR / "train_balanced" / "train_balanced_all.csv"
    val_file_path   = DATA_DIR / "val_balanced"   / "val_balanced_all.csv"
    test_file_path  = DATA_DIR / "test_balanced"  / "test_balanced_all.csv"
    
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
    
    
    # Cap (per MAC) used by the CSV loader; keep consistent across splits for fairness
    MAX_SAMPLES_PER_MAC_TRAIN = 120_274
    MAX_SAMPLES_PER_MAC_VAL   = 25_023
    MAX_SAMPLES_PER_MAC_TEST  = 26_398
    
    # --------------------------
    # TRAIN set (entire CSV)
    # --------------------------
    # NOTE: set_global_seed above already seeds Python/torch. The CSV loader uses random.sample;
    # this makes the uniform sub-sampling per-MAC reproducible.
    train_data, train_labels = process_csv_fixed_id_uniform_sampling_rssi(
        file_path=train_file_path,
        mac_id_list=MAC_ID_LIST,
        max_samples_per_mac=MAX_SAMPLES_PER_MAC_TRAIN
    )
    assert train_data is not None, "No valid TRAIN data found."
    train_labels = train_labels.long()
    
    # CNN view (TRAIN): [N,1,64,2] + mean_norm()
    x_cnn_train = mean_norm(train_data.unsqueeze(1).float())
    cnn_train_ds = TensorDataset(x_cnn_train, train_labels)
    cnn_train_loader = DataLoader(cnn_train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    
    # CLIP view (TRAIN): [N,2,64] (norm/aug in CSIDataset)
    x_clip_train = train_data.permute(0, 2, 1).float()
    clip_train_ds = CSIDataset(x_clip_train, train_labels, normalize=True, augment=True)
    clip_train_ds.train_mode = True  # enable aug in loader built inside train_clip
    
    print(f"[TRAIN CSV] TRAIN={len(train_labels)}")
    
    # --------------------------
    # VAL set (entire CSV)
    # --------------------------
    # Ensure deterministic uniform sub-sampling per MAC for VAL as well
    random.seed(SEED)
    val_data, val_labels = process_csv_fixed_id_uniform_sampling_rssi(
        file_path=val_file_path,
        mac_id_list=MAC_ID_LIST,
        max_samples_per_mac=MAX_SAMPLES_PER_MAC_VAL
    )
    assert val_data is not None, "No valid VAL data found."
    val_labels = val_labels.long()
    
    # CNN view (VAL)
    x_cnn_val = mean_norm(val_data.unsqueeze(1).float())
    cnn_val_ds = TensorDataset(x_cnn_val, val_labels)
    cnn_val_loader = DataLoader(cnn_val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    
    # CLIP view (VAL)
    x_clip_val = val_data.permute(0, 2, 1).float()
    clip_val_ds = CSIDataset(x_clip_val, val_labels, normalize=True, augment=False)
    clip_val_ds.train_mode = False
    
    print(f"[VAL CSV]   VAL={len(val_labels)}")
    
    # --------------------------
    # TEST set (entire CSV) — shared for all models
    # --------------------------
    random.seed(SEED)
    test_data, test_labels = process_csv_fixed_id_uniform_sampling_rssi(
        file_path=test_file_path,
        mac_id_list=MAC_ID_LIST,
        max_samples_per_mac=MAX_SAMPLES_PER_MAC_TEST
    )
    assert test_data is not None and test_labels is not None, "No valid TEST data found."
    test_labels = test_labels.long()
    
    # CNN view (TEST)
    x_cnn_test = mean_norm(test_data.unsqueeze(1).float())
    cnn_test_ds = TensorDataset(x_cnn_test, test_labels)
    cnn_test_loader = DataLoader(cnn_test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    
    # CLIP view (TEST)
    x_clip_test = test_data.permute(0, 2, 1).float()
    clip_test_ds = CSIDataset(x_clip_test, test_labels, normalize=True, augment=False)
    clip_test_ds.train_mode = False
    clip_test_loader = DataLoader(clip_test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"[TEST CSV]  TEST={len(test_labels)} (shared for all models)")


#%% Training the simpleCNN model with training data loaded from previous cell. Testing the cnn-3 model with testing data loaded from previous cell. 


if False:
    print("Starting training the loaded data on simpleCNN model")
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
    train(model=model_1, 
          train_loader=cnn_train_loader, 
          test_loader=cnn_val_loader, 
          criterion=criterion, 
          optimizer=optimizer, 
          num_epochs=num_epochs)
    
    print("Trianing finished on simpleCNN model")
    
    print("Testing the data simpleCNN model on the loaded testing data")
    # Model Testing
    model_1.eval()
    _ = test(model_1, cnn_test_loader)
    
    print("Testing finished on simpleCNN model")
    
#%% Training the CNN_2 model with training data loaded from previous cell. Testing the cnn-3 model with testing data loaded from previous cell.
    
if False:    
    print("Starting training the loaded data on CNN_2 model")
    num_classes = 8
    learning_rate = 0.001
    num_epochs = 10
    # Model setup
    model_1 = CNN_2(num_classes)
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
    
    print("Trianing finished on CNN_2 model")
    
    print("Testing the data CNN_2 model on the loaded testing data")
    # Model Testing
    model_1.eval()
    _ = test(model_1, cnn_test_loader)
    
    print("Testing finished on CNN_2 model")



#%% Training the ResNet50CSI model with training data loaded from previous cell. Testing the cnn-3 model with testing data loaded from previous cell.
if False:
    num_classes = 8
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
        num_epochs=4
    )
    
    # Evaluate on primary test set
    model_3.eval()
    _ = test(model_3, cnn_test_loader)
    
    
#%% Training the CSI_CLIP model with CNN_2 CSI encoder training data loaded from previous cell. Testing the CSI_CLIP model with cnn-3 CSI encoder model with testing data loaded from previous cell.
if False:    
    # Assuming you have CSI_CLIP, CSIEncoder, LabelEmbedder, evaluate_zero_shot, and train(...) from earlier
    num_classes = len(MAC_ID_LIST)
    clip_model = CSI_CLIP(
        csi_encoder=CSIEncoder(in_ch=2, proj_dim=256),
        label_encoder=LabelEmbedder(num_classes=num_classes, dim=256),
    )
    
    trained_clip = train_clip(
        clip_model, clip_train_ds, clip_val_ds,
        epochs=10, batch_size=64, lr=1e-3, wd=1e-4, num_workers=0, device="cuda"
    )
    
    # Final test on the external TEST CSV (same samples as CNN)
    clip_test_acc = evaluate_zero_shot(trained_clip, clip_test_loader, device=("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[CLIP] TEST top-1: {clip_test_acc*100:.2f}%")


#%% Training the CSI_CLIP model with  CSIResNet50 encoder training data loaded from previous cell. Testing CSI_CLIP model with  CSIResNet50 encoder model with testing data loaded from previous cell. 

if False:
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

#%% CSI_CLIP with CNN_2 CSI enoder with MAC hex -> learnable projector 

if False:
    # ======= Reuse EXACT SAME loaders/splits you already created =======
    # clip_train_ds, clip_val_ds, clip_test_loader come from your code.
    
    # Provide MAC → (x,y,z) dict (meters), keys must match MAC_ID_LIST entries:
    mac_to_xyz = {
        "00:FC:BA:38:4B:00": (19.61, 18.60, 4),
        "00:FC:BA:38:4B:01": (19.61, 18.60, 4),
        "00:FC:BA:38:4B:02": (19.61, 18.60, 4),
        "6C:B2:AE:39:1A:A0": (17.83, 11.30, 4),
        "6C:B2:AE:39:1A:A1": (17.83, 11.30, 4),
        "70:0F:6A:DE:EC:A0": (15.66, 11.30, 4),
        "70:0F:6A:DE:EC:A1": (15.66, 11.30, 4),
        "70:0F:6A:DE:EC:A2": (15.66, 11.30, 4),
    }
    
    num_classes = len(MAC_ID_LIST)
    
    # ---------------------------
    # (1) CSI = 2-layer CNN; Label = MAC hex → learnable projector
    # ---------------------------
    model_1 = CSI_CLIP(
        csi_encoder=CSIEncoder(in_ch=2, proj_dim=256),                 # REUSED encoder (CNN) :contentReference[oaicite:7]{index=7}
        label_encoder=LabelHexProjector(MAC_ID_LIST, dim=256, hex_dim=64)  # NEW
    )
    trained_1 = train_clip(model_1, clip_train_ds, clip_val_ds,
                           epochs=10, batch_size=64, lr=1e-3, wd=1e-4,
                           num_workers=0, device=("cuda" if torch.cuda.is_available() else "cpu"))  # REUSED loop :contentReference[oaicite:8]{index=8}
    acc_1 = evaluate_zero_shot(trained_1, clip_test_loader, device=("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"(1) CNN + MAC(hex→learnable) TEST top-1: {acc_1*100:.2f}%")

#%% CSI_CLIP with CNN_2 CSI enoder with MAC hex → learnable  +  fixed location RFF

if False:
    # ---------------------------
    # (2) CSI = 2-layer CNN; Label = MAC hex → learnable  +  fixed location RFF
    # ---------------------------
    model_2 = CSI_CLIP(
        csi_encoder=CSIEncoder(in_ch=2, proj_dim=256),                 # REUSED
        label_encoder=LabelHexPlusLoc(MAC_ID_LIST, mac_to_xyz,
                                      dim=256, hex_dim=64,
                                      lambda_pos=1.0, sigma=1.0,
                                      seed=SEED, coord_scale=1.0)      # NEW
    )
    trained_2 = train_clip(model_2, clip_train_ds, clip_val_ds,
                           epochs=10, batch_size=64, lr=1e-3, wd=1e-4,
                           num_workers=0, device=("cuda" if torch.cuda.is_available() else "cpu"))  # REUSED
    acc_2 = evaluate_zero_shot(trained_2, clip_test_loader, device=("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"(2) CNN + MAC(hex→learnable)+LOC(RFF) TEST top-1: {acc_2*100:.2f}%")

#%% CSI_CLIP with ResNet50 CSI enoder with MAC hex → learnable

if False:
    # ---------------------------
    # (3) CSI = ResNet50; Label = MAC hex → learnable projector
    # ---------------------------
    model_3 = CSI_CLIP(
        csi_encoder=CSIResNet50Encoder(proj_dim=256, pretrained=True), # REUSED encoder (ResNet50) :contentReference[oaicite:9]{index=9}
        label_encoder=LabelHexProjector(MAC_ID_LIST, dim=256, hex_dim=64)  # NEW
    )
    trained_3 = train_clip(model_3, clip_train_ds, clip_val_ds,
                           epochs=10, batch_size=64, lr=1e-3, wd=1e-4,
                           num_workers=0, device=("cuda" if torch.cuda.is_available() else "cpu"))  # REUSED
    acc_3 = evaluate_zero_shot(trained_3, clip_test_loader, device=("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"(3) ResNet50 + MAC(hex→learnable) TEST top-1: {acc_3*100:.2f}%")

#%% CSI_CLIP with ResNet50 CSI enoder with MAC hex → learnable +  fixed location RFF
if False:
    # ---------------------------
    # (4) CSI = ResNet50; Label = MAC hex → learnable  +  fixed location RFF
    # ---------------------------
    model_4 = CSI_CLIP(
        csi_encoder=CSIResNet50Encoder(proj_dim=256, pretrained=True), # REUSED
        label_encoder=LabelHexPlusLoc(MAC_ID_LIST, mac_to_xyz,
                                      dim=256, hex_dim=64,
                                      lambda_pos=1.0, sigma=1.0,
                                      seed=SEED, coord_scale=1.0)      # NEW
    )
    trained_4 = train_clip(model_4, clip_train_ds, clip_val_ds,
                           epochs=10, batch_size=64, lr=1e-3, wd=1e-4,
                           num_workers=0, device=("cuda" if torch.cuda.is_available() else "cpu"))  # REUSED
    acc_4 = evaluate_zero_shot(trained_4, clip_test_loader, device=("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"(4) ResNet50 + MAC(hex→learnable)+LOC(RFF) TEST top-1: {acc_4*100:.2f}%")
#%% OPTUNA HPO for CNN_2, CLIP_CNN_MAC, CLIP_CNN_MAC_LOC, CLIP_RESNET_MAC, CLIP_RESNET_MAC_LOC
if True:
        # Folder where main.py lives (repo root)
    PROJECT_ROOT = Path(__file__).resolve().parent
    
    # HPO output directory: <repo>/hpo_results
    HPO_OUTDIR = PROJECT_ROOT / "hpo_results"
    HPO_OUTDIR.mkdir(parents=True, exist_ok=True)

    SAVE_TRIALS_CSV   = True   # set False if you don’t want CSVs
    SAVE_CHECKPOINTS  = True   # set False if you don’t want .ckpt files

    # ===================== launch studies =====================
    STUDY_TRIALS = 100  # adjust per compute

#%% mac_to_xyz

if True:
    mac_to_xyz = {
        "00:FC:BA:38:4B:00": (19.61, 18.60, 4),
        "00:FC:BA:38:4B:01": (19.61, 18.60, 4),
        "00:FC:BA:38:4B:02": (19.61, 18.60, 4),
        "6C:B2:AE:39:1A:A0": (17.83, 11.30, 4),
        "6C:B2:AE:39:1A:A1": (17.83, 11.30, 4),
        "70:0F:6A:DE:EC:A0": (15.66, 11.30, 4),
        "70:0F:6A:DE:EC:A1": (15.66, 11.30, 4),
        "70:0F:6A:DE:EC:A2": (15.66, 11.30, 4),
    }

#%% Optimzing CNN_2 classifier
if False:
    # study_0: CNN_2 classifier
    print("Now urnning study_0 for CNN_2_CLASSIFIER")

    # Build objective with explicit context (no globals)
    objective_0 = make_objective_cnn2_classifier(
        mac_id_list=MAC_ID_LIST,
        device=device,
        has_datasets=True,                 # set False to reuse fixed loaders
        train_ds=cnn_train_ds,
        val_ds=cnn_val_ds,
        test_ds=cnn_test_ds,
        # fixed_train_loader=cnn_train_loader, fixed_val_loader=cnn_val_loader, fixed_test_loader=cnn_test_loader,
    )

    study_0 = run_study(objective_0, "CNN2_CLASSIFIER", n_trials=STUDY_TRIALS, seed=SEED)

    # save best VAL params right away
    save_study_best(study_0, "CNN2_CLASSIFIER", HPO_OUTDIR, seed=SEED, save_trials_csv=SAVE_TRIALS_CSV)

    # retrain + TEST, then append test result to JSON and save checkpoint
    test_0 = retrain_and_test_cnn2(
        study_0.best_trial.params,
        mac_id_list=MAC_ID_LIST,
        device=device,
        has_datasets=True,
        outdir=HPO_OUTDIR,
        save_checkpoints=SAVE_CHECKPOINTS,
        train_ds=cnn_train_ds, val_ds=cnn_val_ds, test_ds=cnn_test_ds,
    )
    save_study_best(study_0, "CNN2_CLASSIFIER", HPO_OUTDIR, seed=SEED, extra={"final_test_top1": float(test_0)})
    print(f"[study_0] TEST top-1 (best params): {test_0*100:.2f}%")


#%% Optimzing CLIP_CNN_MAC
if True:
    # study_1: CLIP_CNN_MAC
    print("Now urnning study_1 for CLIP_CNN_MAC")

    objective_1 = make_objective_clip_cnn_mac(
        mac_id_list=MAC_ID_LIST,
        device=device,
        clip_train_ds=clip_train_ds,
        clip_val_ds=clip_val_ds,
    )
    study_1 = run_study(objective_1, "CLIP_CNN_MAC", n_trials=STUDY_TRIALS, seed=SEED)
    save_study_best(study_1, "CLIP_CNN_MAC", HPO_OUTDIR, seed=SEED, save_trials_csv=SAVE_TRIALS_CSV)

    test_1 = retrain_and_test(
        study_1.best_trial.params,
        is_resnet=False, use_location=False,
        mac_id_list=MAC_ID_LIST, mac_to_xyz=None,
        seed=SEED, device=device,
        clip_train_ds=clip_train_ds, clip_val_ds=clip_val_ds, clip_test_loader=clip_test_loader,
        outdir=HPO_OUTDIR, save_checkpoints=SAVE_CHECKPOINTS,
    )
    save_study_best(study_1, "CLIP_CNN_MAC", HPO_OUTDIR, seed=SEED, extra={"final_test_top1": float(test_1)})
    print(f"[study_1] TEST top-1 (best params): {test_1*100:.2f}%")


#%% Optimzing CLIP_CNN_MAC_LOC
if True:
    # study_2: CLIP_CNN_MAC_LOC
    print("Now urnning study_2 for CLIP_CNN_MAC_LOC")

    objective_2 = make_objective_clip_cnn_mac_loc(
        mac_id_list=MAC_ID_LIST,
        mac_to_xyz=mac_to_xyz,
        seed=SEED,
        device=device,
        clip_train_ds=clip_train_ds,
        clip_val_ds=clip_val_ds,
    )
    study_2 = run_study(objective_2, "CLIP_CNN_MAC_LOC", n_trials=STUDY_TRIALS, seed=SEED)
    save_study_best(study_2, "CLIP_CNN_MAC_LOC", HPO_OUTDIR, seed=SEED, save_trials_csv=SAVE_TRIALS_CSV)

    test_2 = retrain_and_test(
        study_2.best_trial.params,
        is_resnet=False, use_location=True,
        mac_id_list=MAC_ID_LIST, mac_to_xyz=mac_to_xyz,
        seed=SEED, device=device,
        clip_train_ds=clip_train_ds, clip_val_ds=clip_val_ds, clip_test_loader=clip_test_loader,
        outdir=HPO_OUTDIR, save_checkpoints=SAVE_CHECKPOINTS,
    )
    save_study_best(study_2, "CLIP_CNN_MAC_LOC", HPO_OUTDIR, seed=SEED, extra={"final_test_top1": float(test_2)})
    print(f"[study_2] TEST top-1 (best params): {test_2*100:.2f}%")


#%% Optimzing CLIP_RESNET_MAC
if True:
    # study_3: CLIP_RESNET_MAC
    print("Now urnning study_3 for CLIP_RESNET_MAC")

    objective_3 = make_objective_clip_resnet_mac(
        mac_id_list=MAC_ID_LIST,
        device=device,
        clip_train_ds=clip_train_ds,
        clip_val_ds=clip_val_ds,
    )
    study_3 = run_study(objective_3, "CLIP_RESNET_MAC", n_trials=STUDY_TRIALS, seed=SEED)
    save_study_best(study_3, "CLIP_RESNET_MAC", HPO_OUTDIR, seed=SEED, save_trials_csv=SAVE_TRIALS_CSV)

    test_3 = retrain_and_test(
        study_3.best_trial.params,
        is_resnet=True, use_location=False,
        mac_id_list=MAC_ID_LIST, mac_to_xyz=None,
        seed=SEED, device=device,
        clip_train_ds=clip_train_ds, clip_val_ds=clip_val_ds, clip_test_loader=clip_test_loader,
        outdir=HPO_OUTDIR, save_checkpoints=SAVE_CHECKPOINTS,
    )
    save_study_best(study_3, "CLIP_RESNET_MAC", HPO_OUTDIR, seed=SEED, extra={"final_test_top1": float(test_3)})
    print(f"[study_3] TEST top-1 (best params): {test_3*100:.2f}%")


#%% Optimzing CLIP_RESNET_MAC_LOC
if True:
    # study_4: CLIP_RESNET_MAC_LOC
    print("Now urnning study_4 for CLIP_RESNET_MAC_LOC")

    objective_4 = make_objective_clip_resnet_mac_loc(
        mac_id_list=MAC_ID_LIST,
        mac_to_xyz=mac_to_xyz,
        seed=SEED,
        device=device,
        clip_train_ds=clip_train_ds,
        clip_val_ds=clip_val_ds,
    )
    study_4 = run_study(objective_4, "CLIP_RESNET_MAC_LOC", n_trials=STUDY_TRIALS, seed=SEED)
    save_study_best(study_4, "CLIP_RESNET_MAC_LOC", HPO_OUTDIR, seed=SEED, save_trials_csv=SAVE_TRIALS_CSV)

    test_4 = retrain_and_test(
        study_4.best_trial.params,
        is_resnet=True, use_location=True,
        mac_id_list=MAC_ID_LIST, mac_to_xyz=mac_to_xyz,
        seed=SEED, device=device,
        clip_train_ds=clip_train_ds, clip_val_ds=clip_val_ds, clip_test_loader=clip_test_loader,
        outdir=HPO_OUTDIR, save_checkpoints=SAVE_CHECKPOINTS,
    )
    save_study_best(study_4, "CLIP_RESNET_MAC_LOC", HPO_OUTDIR, seed=SEED, extra={"final_test_top1": float(test_4)})
    print(f"[study_4] TEST top-1 (best params): {test_4*100:.2f}%")


#%% Optimizing RESNET50CSI_CLASSIFIER  (study_5)
if True:
    print("Now urnning study_5 for RESNET50CSI_CLASSIFIER")

    objective_5 = make_objective_resnet50csi_classifier(
        mac_id_list=MAC_ID_LIST,
        device=device,
        has_datasets=True,              # set False to reuse fixed loaders
        train_ds=cnn_train_ds,
        val_ds=cnn_val_ds,
        # fixed_train_loader=cnn_train_loader, fixed_val_loader=cnn_val_loader,
    )
    study_5 = run_study(objective_5, "RESNET50CSI_CLASSIFIER", n_trials=STUDY_TRIALS, seed=SEED)
    save_study_best(study_5, "RESNET50CSI_CLASSIFIER", HPO_OUTDIR, seed=SEED, save_trials_csv=SAVE_TRIALS_CSV)

    test_5 = retrain_and_test_resnet50csi(
        study_5.best_trial.params,
        mac_id_list=MAC_ID_LIST,
        device=device,
        has_datasets=True,
        outdir=HPO_OUTDIR,
        save_checkpoints=SAVE_CHECKPOINTS,
        train_ds=cnn_train_ds, val_ds=cnn_val_ds, test_ds=cnn_test_ds,
        # fixed_train_loader=cnn_train_loader, fixed_val_loader=cnn_val_loader, fixed_test_loader=cnn_test_loader,
    )
    save_study_best(study_5, "RESNET50CSI_CLASSIFIER", HPO_OUTDIR, seed=SEED, extra={"final_test_top1": float(test_5)})
    print(f"[study_5] TEST top-1 (best params): {test_5*100:.2f}%")
