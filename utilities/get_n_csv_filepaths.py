#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 00:29:56 2025

@author: fawaz
"""
import os

def get_n_csv_filepaths(folder_path, n):
    """
    Retrieves the first 'n' CSV file paths from the given folder based on modification date 
    (oldest files first, most recently modified file last) and formats them as a Python list.

    Args:
    folder_path (str): The path to the folder containing CSV files.
    n (int): The number of file paths to retrieve.

    Returns:
    None: Prints the formatted list of file paths.
    """
    
    # Get all CSV files in the folder with their full paths
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
    
    # Sort files based on modification time (oldest first)
    csv_files.sort(key=lambda x: os.path.getmtime(x))

    # Select the first 'n' files
    csv_file_paths = csv_files[:n]
    
    # Format output for easy copy-pasting as a Python list
    if csv_file_paths:
        print("[")
        for i, path in enumerate(csv_file_paths):
            if i < len(csv_file_paths) - 1:
                print(f'    \"{path}\", \\')
            else:
                print(f'    \"{path}\"')
        print("]")