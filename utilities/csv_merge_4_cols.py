# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:41:23 2025

@author: fawaz243
"""
'''
This function is a modified version of the combine_csv_file() function from merge.py
file. This function is deployed to consider the case where there could be four columns
in  the csv files with two addtional columns being rssi and noise floor in recent captured
data.
'''

import os  # Module for handling file paths and directories
import pandas as pd  # Pandas library for working with CSV files

def combine_csv_files_4_cols(file_paths, output_file):
    """
    Combines multiple CSV files into one, handling cases where files have either 2 or 4 columns.
    
    - Reads each CSV file and determines its column count.
    - If the file has 2 columns, it includes both columns.
    - If the file has 4 columns, it includes all four.
    - Skips files with unexpected column counts.
    - Concatenates all valid CSVs and saves them to the output file.

    Args:
        file_paths (list): List of paths to CSV files.
        output_file (str): Path to the output CSV file.

    Returns:
        None
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)  # Get the directory path from the output file path
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it does not exist

    combined_df = pd.DataFrame()  # Initialize an empty DataFrame to store combined data

    for file in file_paths:
        try:
            # Read the CSV file without headers to determine the number of columns
            df = pd.read_csv(file, header=None)
            
            # If the file has 2 columns, use both; if it has 4 columns, use all
            if df.shape[1] == 2:
                df = df.iloc[:, :2]  # Keep both columns
            elif df.shape[1] == 4:
                df = df.iloc[:, :4]  # Keep all four columns
            else:
                print(f"Skipping {file}: Unexpected number of columns ({df.shape[1]})")
                continue  # Skip processing this file

            # Concatenate the valid CSV data with the combined DataFrame
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error processing {file}: {e}")  # Print an error message if a file fails to process

    # Save the combined data to the output file without headers
    combined_df.to_csv(output_file, index=False, header=False)
    print(f"Combined CSV saved to: {output_file}")