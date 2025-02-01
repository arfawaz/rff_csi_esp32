#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:45:55 2025

@author: fawaz
"""

"""
CSI Phase Extraction and Plotting Script

This script processes raw CSI (Channel State Information) data from a CSV file,
extracts the phase values for selected subcarriers, and plots them over time
for each unique MAC ID.

### Functionality:
1. **Load Data:** Reads a CSV file where each row contains a MAC ID and CSI data.
2. **Parse CSI Data:** Extracts In-phase (I) and Quadrature (Q) values, stored as interleaved integers.
3. **Calculate Phase:** Computes the phase angle for user-specified subcarriers using `atan2(Q, I)`.
4. **Convert to Degrees:** Transforms the phase values from radians to degrees for better readability.
5. **Filter by MAC ID:** Processes data separately for each unique MAC address in the dataset.
6. **Plot Phases Over Time:** Generates time-series plots for each MAC ID, visualizing phase variations.

### Arguments:
- **file_path:** Path to the CSV file containing CSI data.
- **subcarrier_numbers:** List of subcarriers to extract and analyze (e.g., `[6, 7]`).
- **time_limit:** Tuple defining the x-axis limit for plotting (e.g., `(100, 200)`).

### Expected CSV Format:
Each row contains:
- `mac_id` (string): Identifier for the transmitting/receiving device.
- `csi_data` (string): Space-separated list of 128 integer values (64 pairs of I/Q components).

### Output:
- Time-series phase plots for each MAC ID, showing phase changes across selected subcarriers.

### Notes:
- Ensure that the CSV file path is correct.
- Modify `subcarrier_numbers` to analyze different subcarriers.
- Use `time_limit = None` if you want the full time range in plots.

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate phase from I and Q values
def calculate_phase(i, q):
    return np.arctan2(q, i)  # atan2 is used to get the phase in radians

# Convert radians to degrees
def radians_to_degrees(radians):
    return np.degrees(radians)

# Load the data into a pandas DataFrame
file_path = file_path = "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/new_esp4.csv"  # Replace with your actual file path

data = pd.read_csv(file_path, header=None, names=["mac_id", "csi_data"])

# Parse the csi_data column
# Each row's CSI data is a string of 128 space-separated integers
data['csi_data'] = data['csi_data'].apply(lambda x: list(map(int, x.split())))

# Extract phases for user-defined subcarriers
def extract_phases(csi_data, subcarrier_numbers):
    phases = []
    for n in subcarrier_numbers:
        i = csi_data[2 * n]     # I value for subcarrier n
        q = csi_data[2 * n + 1] # Q value for subcarrier n
        phase = calculate_phase(i, q)
        phases.append(radians_to_degrees(phase))  # Convert to degrees
    return phases

# Set the list of subcarriers to plot (example: first 3 subcarriers)
subcarrier_numbers = [6,7] # Modify this list as needed

# Time axis limit (can be None for no limit)
time_limit = (100, 200)  # Example: Limit the time axis from 0 to 100 (modify as needed)

# Apply the phase extraction to each row for the selected subcarriers
data['phases'] = data['csi_data'].apply(lambda x: extract_phases(x, subcarrier_numbers))

# Plot the phases for each mac_id
mac_ids = data['mac_id'].unique()

for mac_id in mac_ids:
    # Filter the data for the current mac_id
    mac_data = data[data['mac_id'] == mac_id]

    # Create a new figure for each mac_id
    plt.figure(figsize=(10, 6))

    # Plot the phase of the selected subcarriers
    for i, subcarrier in enumerate(subcarrier_numbers):
        plt.plot(mac_data.index, mac_data['phases'].apply(lambda x: x[i]), label=f'Subcarrier {subcarrier + 1}')

    # Add labels and title
    plt.title(f"Phase vs Time (in Degrees) for MAC ID: {mac_id}")
    plt.xlabel('Time')
    plt.ylabel('Phase (degrees)')
    plt.legend()
    plt.grid(True)

    # Set the time axis limit if specified
    if time_limit is not None:
        plt.xlim(time_limit)

    # Show the plot
    plt.show()

