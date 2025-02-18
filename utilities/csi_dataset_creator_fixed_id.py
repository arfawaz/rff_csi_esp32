#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:12:13 2025

@author: fawaz
"""

import csv
import torch

def parse_csi_data(csi_row):
    """
    Parses a single row of CSI data into a 64x2 PyTorch tensor.
    """
    csi_values = csi_row.split()
    if len(csi_values) != 128:
        return None  
    csi_tensor = []
    for i in range(0, 128, 2):
        try:
            magnitude = float(csi_values[i])
            angle = float(csi_values[i + 1])
            csi_tensor.append([magnitude, angle])
        except ValueError:
            return None
    return torch.tensor(csi_tensor)

def process_csv_fixed_id(file_path, mac_id_list, max_samples_per_mac=50000):
    """
    Processes a CSV file to extract CSI data for specific MAC addresses and assigns labels based on their order in mac_id_list.
    """
    data = []
    labels = []
    mac_id_sample_count = {mac: 0 for mac in mac_id_list}  # Track samples for each MAC ID
    mac_id_to_label = {mac: i for i, mac in enumerate(mac_id_list)}  # Assign labels based on order
    
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) != 2:
                continue  
            current_mac_id, csi_row = row  
            if current_mac_id not in mac_id_list:
                continue  # Skip MAC IDs not in the specified list
            
            csi_tensor = parse_csi_data(csi_row)
            if csi_tensor is not None and mac_id_sample_count[current_mac_id] < max_samples_per_mac:
                data.append(csi_tensor)
                labels.append(mac_id_to_label[current_mac_id])
                mac_id_sample_count[current_mac_id] += 1  
    
    if data:
        data_ = torch.stack(data)
        labels_ = torch.tensor(labels, dtype=torch.long)
        return data_, labels_
    else:
        return None, None  # Return None if no valid data was processed

