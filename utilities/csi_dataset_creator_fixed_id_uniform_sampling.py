#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:31:10 2025

@author: fawaz
"""

import csv
import torch
import random

def parse_csi_data(csi_row):
    """
    Parses a single row of CSI data into a 64x2 PyTorch tensor.
    """
    csi_values = csi_row.split()
    if len(csi_values) != 128:
        return None  # Skip invalid CSI rows
    csi_tensor = []
    for i in range(0, 128, 2):
        try:
            magnitude = float(csi_values[i])
            angle = float(csi_values[i + 1])
            csi_tensor.append([magnitude, angle])
        except ValueError:
            return None  # Skip rows with invalid numeric values
    return torch.tensor(csi_tensor)

def process_csv_fixed_id_uniform_sampling(file_path, mac_id_list, max_samples_per_mac=50000):
    """
    Processes a CSV file to extract CSI data for specific MAC addresses and assigns labels based on their order in mac_id_list.
    Instead of selecting the first max_samples_per_mac entries, this function selects uniformly from all available entries.
    """
    mac_entries = {mac: [] for mac in mac_id_list}  # Store all CSI data for each MAC
    
    # Read CSV file and collect all valid CSI entries for each MAC
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) != 2:
                continue  # Skip invalid rows
            current_mac_id, csi_row = row  
            if current_mac_id not in mac_id_list:
                continue  # Skip MACs not in the specified list
            
            csi_tensor = parse_csi_data(csi_row)
            if csi_tensor is not None:
                mac_entries[current_mac_id].append(csi_tensor)  # Store valid CSI tensor
    
    # Randomly select up to max_samples_per_mac for each MAC
    data = []
    labels = []
    mac_id_to_label = {mac: i for i, mac in enumerate(mac_id_list)}  # Assign labels based on order

    for mac, entries in mac_entries.items():
        sample_size = min(len(entries), max_samples_per_mac)
        sampled_entries = random.sample(entries, sample_size)  # Uniform random selection

        data.extend(sampled_entries)
        labels.extend([mac_id_to_label[mac]] * sample_size)

    if data:
        data_ = torch.stack(data)
        labels_ = torch.tensor(labels, dtype=torch.long)
        return data_, labels_
    else:
        return None, None  # Return None if no valid data was processed
