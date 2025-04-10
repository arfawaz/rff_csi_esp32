# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 19:54:53 2025

@author: fawaz243
"""

import csv  # Importing csv module to read CSV files
from collections import defaultdict  # defaultdict helps to initialize dictionary keys with default values

def count_target_macid_occurrences(csv_path, target_macids):
    """
    Count the number of times each MAC ID from a given list appears in a CSV file.
    Only rows with valid CSI data (128 values) are considered.
    
    Parameters:
    - csv_path (str): File path to the CSV.
    - target_macids (list of str): List of MAC addresses to track.

    Returns:
    - None (prints the count of each MAC ID from the list)
    """

    # Initialize a dictionary to count occurrences of each MAC ID
    # Automatically sets the initial count to 0 for each new MAC ID
    macid_counts = defaultdict(int)

    # Open the CSV file for reading
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)  # Create a CSV reader object

        # Loop over each row in the CSV
        for row in reader:
            if not row:
                continue  # Skip completely empty rows

            # The first column always contains the MAC ID
            macid = row[0].strip()  # Remove any leading/trailing whitespace

            # Check the number of columns in the row
            # Expected formats:
            #   Format 1: [macid, csi_data]           → len(row) == 2
            #   Format 2: [macid, noise, rssi, csi]   → len(row) == 4
            if len(row) == 2:
                # CSI data is in the second column
                csi_str = row[1].strip()
            elif len(row) == 4:
                # CSI data is in the fourth column
                csi_str = row[3].strip()
            else:
                # Skip any row that doesn't match the expected 2 or 4 column format
                continue

            # Split the CSI string into individual values using space as the delimiter
            csi_values = csi_str.split()

            # Skip rows that don't contain exactly 128 CSI values
            if len(csi_values) != 128:
                continue

            # If the current MAC ID is one of the target MAC IDs, increment its count
            if macid in target_macids:
                macid_counts[macid] += 1

    # After processing all rows, print the counts of each MAC ID from the list
    for mac in target_macids:
        print(f"{mac}: {macid_counts[mac]}")  # If a MAC ID wasn't found, its count will still be 0