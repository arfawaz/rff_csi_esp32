#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:39:28 2025

@author: fawaz
"""
import os
import pandas as pd

def combine_csv_files(file_paths, output_file):
    """
    Combines multiple CSV files into one.

    Args:
        file_paths (list): List of paths to CSV files.
        output_file (str): Path to the output CSV file.

    Returns:
        None
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    combined_df = pd.DataFrame()

    for file in file_paths:
        try:
            # Read CSV without headers and only take the first two columns
            df = pd.read_csv(file, header=None, usecols=[0, 1])
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Save the combined data without headers
    combined_df.to_csv(output_file, index=False, header=False)
    print(f"Combined CSV saved to: {output_file}")

