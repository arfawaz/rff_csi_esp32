# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 22:53:24 2025

@author: fawaz243
"""

import csv
import random
import os
import re
from collections import defaultdict

'''
The purpose of this set of functions is to accept a default dict which has the
percentage values for each position and macid combination. It then uniformly 
samples rows from multiple csv files, with position incoporated in the file name,
which is passed as a parameter. It then creates new csv files for each postion
with rows of csi values sampled according to the percentage value passed to the
function as one of the input parameters. This is used to create test set from
a test csv file such that the macid vs postion total created by sampling the 
test csv file will now have a distribution same as the traning data whose percentage
value is what we pass as one of the inputs to the function.
'''

def extract_position_from_filename(filepath):
    """
    Extract the position identifier from a file path (e.g., 'p4', 'p5').

    Parameters:
        filepath (str): Full path to the CSV file.

    Returns:
        str: Position string extracted from filename, e.g., 'p4'. If not found, returns 'unknown'.
    """
    filename = os.path.basename(filepath)
    match = re.search(r'(p\d+)', filename)  # Matches strings like 'p4'
    return match.group(1) if match else "unknown"


def sample_macid_rows(csv_path, target_macids, adjusted_counts, position, output_folder):
    """
    Sample valid CSI rows from a CSV file based on adjusted sample counts per MAC ID.

    Parameters:
        csv_path (str): Path to input CSV file containing CSI data.
        target_macids (List[str]): List of MAC IDs to include in sampling.
        adjusted_counts (Dict[str, Dict[str, int]]): A nested dictionary containing the number of
                                                     rows to sample for each MAC ID per position.
                                                     Format: adjusted_counts[macid][position] = count
        position (str): Position identifier string (e.g., 'p4', 'p5') for the current file.
        output_folder (str): Path to folder where sampled CSV files will be saved.

    Returns:
        None (writes output to disk).
    """
    macid_rows = defaultdict(list)  # Store valid CSI rows grouped by MAC ID

    # Read and filter the CSV
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue  # Skip empty rows

            macid = row[0].strip()

            # Depending on row format, extract the CSI string
            if len(row) == 2:
                csi_str = row[1].strip()
            elif len(row) == 4:
                csi_str = row[3].strip()
            else:
                continue  # Skip malformed rows

            if len(csi_str.split()) != 128:
                continue  # Skip invalid CSI rows (must contain 128 values)

            # If MAC ID is one of the target MACs, store the row
            if macid in target_macids:
                macid_rows[macid].append(row)

    sampled_rows = []  # To store the final sampled rows

    # Perform random sampling based on adjusted count
    for macid in target_macids:
        if macid in macid_rows:
            num_samples = int(adjusted_counts[macid][position])
            print(f'num_samples is: {num_samples}')

            if len(macid_rows[macid]) >= num_samples:
                sampled_rows.extend(random.sample(macid_rows[macid], num_samples))
            else:
                sampled_rows.extend(macid_rows[macid])  # Not enough rows; take all available

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the sampled rows to a new CSV file
    new_file_name = os.path.join(output_folder, f"sampled_{position}.csv")

    with open(new_file_name, 'w', newline='') as new_file:
        writer = csv.writer(new_file)
        writer.writerows(sampled_rows)

    print(f"Sampled data for position {position} saved in {new_file_name}")


def create_sampled_csv_files_from_table(mac_ids, file_paths, adjusted_counts, output_folder):
    """
    Wrapper function that loops through multiple input CSV files (each representing a position)
    and samples CSI rows for each MAC ID based on provided adjusted counts.

    Parameters:
        mac_ids (List[str]): List of MAC addresses to sample for.
        file_paths (List[str]): List of CSV file paths for different positions.
        adjusted_counts (Dict[str, Dict[str, int]]): Sampling count per MAC per position.
        output_folder (str): Directory where sampled files will be saved.

    Returns:
        None
    """
    for file_path in file_paths:
        position = extract_position_from_filename(file_path)
        print(f"position is: {position}")
        sample_macid_rows(file_path, mac_ids, adjusted_counts, position, output_folder)
