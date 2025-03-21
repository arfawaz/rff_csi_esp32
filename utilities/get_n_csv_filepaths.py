#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 00:29:56 2025

@author: fawaz
"""

def get_n_csv_filepaths(folder_path, n):
    """
    Retrieves the first 'n' CSV file paths from the given folder and formats them as a Python list.
    
    Args:
    folder_path (str): The path to the folder containing CSV files.
    n (int): The number of file paths to retrieve.

    Returns:
    None: Prints the formatted list of file paths.
    """
    
    # Get a sorted list of all CSV files in the folder
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
    
    # Get the full file paths
    csv_file_paths = [os.path.join(folder_path, f) for f in csv_files[:n]]
    
    # Format output for easy copy-pasting as a Python list
    if csv_file_paths:
        print("[")
        for i, path in enumerate(csv_file_paths):
            if i < len(csv_file_paths) - 1:
                print(f'    "{path}", \\')
            else:
                print(f'    "{path}"')
        print("]")