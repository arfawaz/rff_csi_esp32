#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 20:38:02 2025

@author: fawaz
"""

import os
import shutil
from pathlib import Path

def copy_csv_with_serial(source_folder, destination_folder):
    """
    Copies all .csv files from source_folder to destination_folder.
    Each file is renamed with a serial number prefix based on its last modified time
    (older files get lower serial numbers). The original files are not modified.
    """

    # Create the destination folder if it does not exist
    os.makedirs(destination_folder, exist_ok=True)

    # Get a list of all .csv files in the source folder
    # Pathlib's glob method returns all files that match the pattern "*.csv"
    # We ensure they are actually files (not directories)
    csv_files = [f for f in Path(source_folder).glob("*.csv") if f.is_file()]

    # Sort the files based on their last modified time (ascending)
    # This means older files get smaller serial numbers
    csv_files.sort(key=lambda f: f.stat().st_mtime)

    # Enumerate through the sorted list starting from 1
    for idx, file in enumerate(csv_files, start=1):
        # Construct the new file name with serial number and underscore prefix
        new_name = f"{idx}_{file.name}"

        # Define the destination file path using the new name
        dest_path = Path(destination_folder) / new_name

        # Copy the file to the destination folder with the new name
        shutil.copy(file, dest_path)

    # Print a message indicating completion
    print(f"Copied {len(csv_files)} CSV files to '{destination_folder}' with serial prefixes.")

