#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 16:53:25 2025

@author: fawaz
"""

'''
This function is used to capture the csi data written on the standard input (displayed on terminal)
by the ESP32 configured either as an ap or sta. This function can be called from the command
line by piping the standard output to the this python script.

'''

## PYTHON IMPORTS

import csv  # Module for working with CSV files
import sys  # Module for system-specific parameters and functions
import re  # Module for regular expression operations


## parse_and_write_to_csv fucntion 
def parse_and_write_to_csv(input_line):
    """
    This function processes a single input line, extracts the MAC address and CSI data using a regex,
    and writes them into a CSV file.
    """
    try:
        # Decode the input binary line to a UTF-8 string, ignoring any decoding errors
        line = input_line.decode('utf-8', errors='ignore')
        print(f"Processing line: {line.strip()}")  # Print the line being processed for debugging

        # Use a regular expression to extract the MAC address and CSI data from the input line.
        # Explanation of the regex:
        # - `CSI_DATA,[^,]+,`: Matches the prefix "CSI_DATA," followed by any non-comma characters and a comma.
        # - `([0-9A-F:]+)`: Captures the MAC address, which is a sequence of hexadecimal characters and colons.
        # - `[^,]+,`: Matches additional fields before CSI data, ending in a comma.
        # - `\d+,\d+`: Matches two integer fields (e.g., channel, RSSI) separated by commas.
        # - `.*?\[(.*?)\]`: Matches the CSI data inside square brackets and captures it.
        match = re.match(r'CSI_DATA,[^,]+,([0-9A-F:]+),[^,]+,\d+,\d+.*?\[(.*?)\]', line, re.IGNORECASE)

        if match:  # Check if the regex matched the line format
            # Extract the MAC address and CSI data from the regex groups
            mac_address = match.group(1)
            csi_data = match.group(2)

            # Specify the output CSV file where data will be stored
            csv_filename = "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/esp32/esp32_csi_tool/ESP32-CSI-Tool/active_ap/custom_data_collection/all_macid.csv"

            # Open the CSV file in append mode ('a') and write the extracted data
            with open(csv_filename, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)  # Create a CSV writer object
                # Write the MAC address and CSI data as a row in the CSV file
                csvwriter.writerow([mac_address, csi_data])
            print(f"Data written to {csv_filename}")  # Debugging output indicating successful write

        else:
            # Print an error message if the line does not match the expected format
            print("Invalid data format")  # Debugging output for invalid data

    except Exception as e:
        # Catch and print any exceptions that occur during processing
        print(f"Error processing line: {e}")

## Read input lines from standard input (binary mode) to handle potentially non-UTF-8 data.
with sys.stdin.buffer as binary_input:
    for binary_line in binary_input:
        # Process each binary line by passing it to the `parse_and_write_to_csv` function
        parse_and_write_to_csv(binary_line)
