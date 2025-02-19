#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:21:06 2025

@author: fawaz
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters for uniform distribution
a, b = 0, 1  # Uniform U(0,1)

# Number of samples and sample size
num_samples = 10000  # Total samples to draw
sample_size = 30  # Sample size per iteration

# Generate sample means
sample_means = [np.mean(np.random.uniform(a, b, sample_size)) for _ in range(num_samples)]

# Plot the histogram of sample means
sns.histplot(sample_means, bins=50, kde=True, color="blue")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.title(f"Central Limit Theorem Demonstration (Uniform {a, b}, n={sample_size})")
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the range for x and y
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

# Create a meshgrid
X, Y = np.meshgrid(x, y)

# Compute Z = x^2 * y
Z = (X**2) * Y

# Create the 3D figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)

# Labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Plot of Z = x^2 * y')

plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# Define the mean and covariance matrix
mean = [0, 0]  # Mean (center) at (0,0)
cov = [[1, 0.5], [0.5, 1]]  # Covariance matrix

# Create grid and multivariate normal distribution
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Compute the Bivariate Gaussian distribution
rv = multivariate_normal(mean, cov)
Z = rv.pdf(pos)

# Create the figure
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

# 3D surface plot
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)

# Projection on X-Z plane (Y=0)
ax.contour(X, Y, Z, zdir='y', offset=-3.5, cmap='plasma')

# Projection on Y-Z plane (X=0)
ax.contour(X, Y, Z, zdir='x', offset=-3.5, cmap='magma')

# Labels and title
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Probability Density")
ax.set_title("Bivariate Gaussian Distribution with X-Z and Y-Z Projections")

# Set axis limits
ax.set_xlim([-3.5, 3])
ax.set_ylim([-3.5, 3])
ax.set_zlim([0, np.max(Z)])

plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# Define the mean and covariance matrix
mean = [0, 0]  # Mean (center at origin)
cov = np.array([[2, 1], [1, 1]])  # Covariance matrix

# Eigen decomposition of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# Create grid in the original coordinate system
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Stack to form coordinate pairs
pos = np.dstack((X, Y))

# Compute the Bivariate Gaussian distribution
rv = multivariate_normal(mean, cov)
Z = rv.pdf(pos)

# Transform the coordinate system using eigenvectors
new_coords = np.dot(eigenvectors, np.array([X.flatten(), Y.flatten()]))
X_prime, Y_prime = new_coords[0].reshape(X.shape), new_coords[1].reshape(Y.shape)

# Create the figure
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

# 3D surface plot in the eigenvector-aligned coordinate system
ax.plot_surface(X_prime, Y_prime, Z, cmap='viridis', edgecolor='k', alpha=0.8)

# Projection on X'-Z plane
ax.contour(X_prime, Y_prime, Z, zdir='y', offset=-3.5, cmap='plasma')

# Projection on Y'-Z plane
ax.contour(X_prime, Y_prime, Z, zdir='x', offset=-3.5, cmap='magma')

# Labels and title
ax.set_xlabel("Eigenvector X'-axis")
ax.set_ylabel("Eigenvector Y'-axis")
ax.set_zlabel("Probability Density")
ax.set_title("Bivariate Gaussian Aligned to Eigenvector Axes")

# Set axis limits
ax.set_xlim([-3.5, 3])
ax.set_ylim([-3.5, 3])
ax.set_zlim([0, np.max(Z)])

plt.show()

#%%

import numpy as np

# Generate a random 16x16 integer matrix
A = np.random.randint(-10, 10, (16, 16))

# Make it symmetric
symmetric_matrix = (A + A.T) // 2  # Ensure integer symmetry

# Print the matrix
print(symmetric_matrix)

#%%

import numpy as np

# Generate a random 16x16 integer symmetric matrix
A = np.random.randint(-10, 10, (16, 16))
symmetric_matrix = (A + A.T) // 2  # Ensure symmetry

# Extract four 4x4 submatrices
M1 = symmetric_matrix[:4, :4]   # Top-left
M2 = symmetric_matrix[:4, 4:8]  # Top-right
M3 = symmetric_matrix[4:8, :4]  # Bottom-left
M4 = symmetric_matrix[4:8, 4:8] # Bottom-right

# Print results
print("Full 16x16 Symmetric Matrix:\n", symmetric_matrix)
print("\nTop-left (M1):\n", M1)
print("\nTop-right (M2):\n", M2)
print("\nBottom-left (M3):\n", M3)
print("\nBottom-right (M4):\n", M4)


#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt

def parse_csi_csv(file_path, target_mac, subcarriers, window_size=5, start_time=0, end_time=None):
    """
    Parses the CSI CSV file, filters data for a specific MAC address, 
    extracts amplitude values for selected subcarriers, applies noise filtering,
    and plots amplitude vs. time for a given range.

    :param file_path: Path to the CSV file
    :param target_mac: MAC address to filter
    :param subcarriers: List of subcarrier indices (1 to 64)
    :param window_size: Moving average filter window size
    :param start_time: Start index for plotting
    :param end_time: End index for plotting (None means full range)
    """

    # Read the CSV file
    data = []
    timestamps = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            mac = parts[0].strip()
            
            if mac == target_mac:
                csi_values = list(map(int, parts[1].split()))
                
                # Extract magnitude values (even indices only)
                amplitudes = np.array(csi_values[::2])  # Magnitudes only
                
                # Store selected subcarrier amplitudes
                selected_amplitudes = amplitudes[np.array(subcarriers) - 1]  # Convert 1-based to 0-based index
                
                data.append(selected_amplitudes)
                timestamps.append(len(data))  # Using row index as time for simplicity

    if not data:
        print("No data found for MAC:", target_mac)
        return
    
    data = np.array(data)  # Convert list to NumPy array

    # Apply noise filtering (median filter)
    filtered_data = np.apply_along_axis(lambda x: medfilt(x, kernel_size=window_size), axis=0, arr=data)

    # Define time range
    if end_time is None:
        end_time = len(filtered_data)

    if start_time < 0 or end_time > len(filtered_data) or start_time >= end_time:
        print(f"Invalid time range: {start_time} to {end_time}")
        return

    # Plot results in selected range
    plt.figure(figsize=(12, 6))
    
    for i, subcarrier in enumerate(subcarriers):
        plt.plot(timestamps[start_time:end_time], filtered_data[start_time:end_time, i], label=f'Subcarrier {subcarrier}')
    
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title(f"CSI Amplitude vs. Time for MAC {target_mac} ({start_time} to {end_time})")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
file_path = "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_5aps_fixedpos_16_feb_3.csv"  # Update with actual CSV file path
target_mac = "8C:4F:00:3C:BF:4D"
selected_subcarriers = [50,40]  # Select subcarriers of interest
start_sample = 400
end_sample = 800

parse_csi_csv(file_path, target_mac, selected_subcarriers, start_time=start_sample, end_time=end_sample)




#%%

from mac_id_counter import count_mac_occurrences

file_path = "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_18_feb_1.csv"

count_mac_occurrences(file_path, 5)


#%%

import csv
from collections import Counter

def count_common_mac_occurrences(list_of_file_paths, number_of_top_mac_ids, minimum_number_of_samples=None):
    mac_counters = []  # List to store MAC address counters for each file
    total_entries_per_file = []
    
    # Read each file and count MAC address occurrences
    for csv_file in list_of_file_paths:
        mac_counter = Counter()
        total_entries = 0
        
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            
            for row in reader:
                if len(row) < 2:
                    continue  # Skip invalid rows
                mac_address = row[0].strip()
                mac_counter[mac_address] += 1
                total_entries += 1
        
        mac_counters.append(mac_counter)
        total_entries_per_file.append(total_entries)
    
    # Find MAC addresses common to all files
    common_mac_ids = set(mac_counters[0].keys())
    for mac_counter in mac_counters[1:]:
        common_mac_ids.intersection_update(mac_counter.keys())
    
    if not common_mac_ids:
        print("No common MAC IDs found across all files.")
        return
    
    # Count occurrences of each common MAC ID across files
    common_mac_counts = {}
    for mac in common_mac_ids:
        common_mac_counts[mac] = min(mac_counter[mac] for mac_counter in mac_counters)
    
    # Sort MAC IDs based on the minimum count across files
    sorted_common_macs = sorted(common_mac_counts.items(), key=lambda x: x[1], reverse=True)
    
    if minimum_number_of_samples is None:
        # Select the top `number_of_top_mac_ids` MAC IDs (or all if fewer than required)
        selected_macs = sorted_common_macs[:number_of_top_mac_ids]
        
        print(f"\nCommon Top {len(selected_macs)} MAC Addresses (across all files):")
        for mac, count in selected_macs:
            print(f"{mac}: {count} samples")
        
        if selected_macs:
            least_mac, least_count = selected_macs[-1]
            print(f"\nMAC ID with the least occurrence in the common list: {least_mac} ({least_count} samples)")
    else:
        # Filter MAC IDs that meet the minimum number of samples requirement
        filtered_macs = [(mac, count) for mac, count in sorted_common_macs if count >= minimum_number_of_samples]
        
        if len(filtered_macs) < number_of_top_mac_ids:
            print("The required number of top MAC IDs or minimum number of samples is not met.")
        else:
            print(f"\nMAC IDs meeting minimum {minimum_number_of_samples} samples:")
            for mac, count in filtered_macs:
                print(f"{mac}: {count} samples")
            
            if filtered_macs:
                least_mac, least_count = min(filtered_macs, key=lambda x: x[1])
                print(f"\nMAC ID with the least samples (meeting criteria): {least_mac} ({least_count} samples)")

# Example usage
list_of_files = ["/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_1.csv", \
                 "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_2.csv", \
                "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_3.csv"]
number_of_top_mac_ids = 30
minimum_samples = 121  # Set an integer value if filtering by sample count is needed
count_common_mac_occurrences(list_of_files, number_of_top_mac_ids, minimum_samples)



#%%

import csv
from collections import Counter

def count_common_mac_occurrences(list_of_file_paths, number_of_top_mac_ids, minimum_number_of_samples=None):
    mac_counters = []  # List to store MAC counters for each file
    total_entries = []  # List to store total MAC entries per file
    
    # Read each CSV file and count occurrences of MAC addresses
    for csv_file in list_of_file_paths:
        mac_counter = Counter()
        total_entries_count = 0
        
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) < 2:
                    continue  # Skip invalid rows
                mac_address = row[0].strip()
                mac_counter[mac_address] += 1
                total_entries_count += 1
        
        mac_counters.append(mac_counter)
        total_entries.append(total_entries_count)
    
    # Find MAC addresses common to all files
    common_mac_ids = set(mac_counters[0].keys())
    for mac_counter in mac_counters[1:]:
        common_mac_ids.intersection_update(mac_counter.keys())
    
    if not common_mac_ids:
        print("No common MAC addresses found across all files.")
        return
    
    # Count occurrences of each common MAC address across all files
    common_mac_counts = {mac: min(counter[mac] for counter in mac_counters) for mac in common_mac_ids}
    sorted_common_mac_counts = sorted(common_mac_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Case 1: When minimum_number_of_samples is None
    if minimum_number_of_samples is None:
        top_mac_ids = sorted_common_mac_counts[:number_of_top_mac_ids]
        
        if len(common_mac_ids) < number_of_top_mac_ids:
            print("\nTotal number of common MAC IDs is less than the requested top count.")
        
        # Display the top common MAC addresses and their counts
        print("\nTop Common MAC Addresses:")
        for mac, count in top_mac_ids:
            print(f'{mac}: {count}')
        
        # Print MAC IDs in required list format
        print("\n[")
        for i, (mac, _) in enumerate(top_mac_ids):
            if i == len(top_mac_ids) - 1:
                print(f'"{mac}"')  # Last entry without trailing comma
            else:
                print(f'"{mac}", \\')
        print("]")
        
        # Find MAC ID with the least occurrences
        least_mac, least_count = min(top_mac_ids, key=lambda x: x[1], default=(None, None))
        print(f"\nMAC ID with the least occurrences in the top list: {least_mac} ({least_count} samples)")
    
    # Case 2: When minimum_number_of_samples is an integer
    else:
        filtered_mac_ids = [(mac, count) for mac, count in sorted_common_mac_counts if count >= minimum_number_of_samples]
        
        if len(filtered_mac_ids) < number_of_top_mac_ids:
            print("\nThe required number_of_top_mac_ids with minimum_number_of_samples is not met.")
        else:
            print("\nMAC Addresses with at least minimum_number_of_samples:")
            for mac, count in filtered_mac_ids[:number_of_top_mac_ids]:
                print(f'{mac}: {count}')
        
        # Print MAC IDs in required list format
        print("\n[")
        for i, (mac, _) in enumerate(filtered_mac_ids[:number_of_top_mac_ids]):
            if i == len(filtered_mac_ids[:number_of_top_mac_ids]) - 1:
                print(f'"{mac}"')  # Last entry without trailing comma
            else:
                print(f'"{mac}", \\')
        print("]")
        
        if filtered_mac_ids:
            least_mac, least_count = min(filtered_mac_ids, key=lambda x: x[1])
            print(f"\nMAC ID with the least occurrences in the filtered list: {least_mac} ({least_count} samples)")
        else:
            print("\nNo MAC IDs meet the minimum_number_of_samples requirement.")






# Example usage
list_of_files = ["/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_1.csv", \
                 "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_2.csv", \
                "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_3.csv"]
number_of_top_mac_ids = 10
minimum_samples = None  # Set an integer value if filtering by sample count is needed
count_common_mac_occurrences(list_of_files, number_of_top_mac_ids, minimum_samples)



#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:41:20 2025

@author: fawaz
"""

import csv
from collections import Counter

def count_mac_occurrences(csv_file, n):
    mac_counter = Counter()
    total_entries = 0
    
    # Read the CSV file
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        
        for row in reader:
            if len(row) < 2:
                continue  # Skip invalid rows
            mac_address = row[0].strip()
            mac_counter[mac_address] += 1
            total_entries += 1
    
    # Sort occurrences in decreasing order
    sorted_mac_counts = sorted(mac_counter.items(), key=lambda x: x[1], reverse=True)
    
    # Print occurrences for each unique MAC ID
    print("MAC Address Occurrences (Sorted by Count):")
    #for mac, count in sorted_mac_counts:
        #print(f"{mac}: {count}")
    
    # Print total count of unique MAC addresses
    print("\nTotal Unique MAC IDs:", len(mac_counter))
    print("Total Entries:", total_entries)
    
    # Extract the first n MAC addresses
    top_n_macs = sorted_mac_counts[:n]
    
    # Print the first n MAC addresses in the required format
    print(f"\nTop {n} MAC Addresses:")
    for i, (mac, count) in enumerate(top_n_macs):
        if i == len(top_n_macs) - 1:
            print(f'"{mac}"')  # Last entry without trailing comma
        else:
            print(f'"{mac}", \\')
    
    # Print the count for the least number in these n MAC IDs
    if top_n_macs:
        least_count = top_n_macs[-1][1]
        print(f"\nCount for the least number in these {n} MAC IDs: {least_count}")

    # **NEW PRINT STATEMENT: Print each MAC with its count**
    print(f"\nTop {n} MAC Addresses with Counts:")
    for mac, count in top_n_macs:
        print(f"{mac}: {count}")






file_path = "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_3.csv"

count_mac_occurrences(file_path, 5)


#%%

import csv
from collections import Counter

def count_mac_occurrences(list_of_file_paths, number_of_top_mac_ids, minimum_number_of_samples=None):
    mac_counter = Counter()
    
    # Read each file and count occurrences
    for file_path in list_of_file_paths:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) < 2:
                    continue  # Skip invalid rows
                mac_address = row[0].strip()
                mac_counter[mac_address] += 1
    
    # Find MAC addresses common to all files
    sorted_mac_counts = sorted(mac_counter.items(), key=lambda x: x[1], reverse=True)
    common_macs = [mac for mac, count in sorted_mac_counts]
    
    # Filter by minimum number of samples if provided
    if minimum_number_of_samples is not None:
        common_macs = [(mac, count) for mac, count in sorted_mac_counts if count >= minimum_number_of_samples]
        if not common_macs or len(common_macs) < number_of_top_mac_ids:
            print("No MAC IDs meet the minimum sample requirement or top N requirement.")
            return
    
    # Select the top N MAC IDs
    top_macs = common_macs[:number_of_top_mac_ids]
    
    # Determine the least occurring MAC in the top N
    if top_macs:
        least_common_mac, least_count = top_macs[-1]
        print(f"MAC Address with least occurrences in top {number_of_top_mac_ids}: {least_common_mac} ({least_count} times)")
    
    # Print the formatted MAC ID list
    print("\nFormatted MAC ID List:")
    for i, (mac, _) in enumerate(top_macs):
        if i == len(top_macs) - 1:
            print(f'"{mac}"')
        else:
            print(f'"{mac}", \\')
    
    # Print occurrences for each MAC ID
    print("\nMAC Address Occurrences:")
    for mac, count in top_macs:
        print(f"{mac}: {count}")






# Example usage
list_of_files = ["/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_1.csv", \
                 "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_2.csv", \
                "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_3.csv"]
number_of_top_mac_ids = 10
minimum_samples = None  # Set an integer value if filtering by sample count is needed
count_common_mac_occurrences(list_of_files, number_of_top_mac_ids, minimum_samples)


#%%

import csv
from collections import Counter

def count_common_mac_occurrences(list_of_file_paths, number_of_top_mac_ids, minimum_number_of_samples=None):
    """
   This function reads multiple CSV files, counts MAC address occurrences across all files,
   and finds the top N common MAC addresses based on total occurrences.

   Parameters:
   - list_of_file_paths (list): List of file paths to CSV files.
   - number_of_top_mac_ids (int): Number of top MAC addresses to display.
   - minimum_number_of_samples (int or None): Minimum occurrences required for MACs (optional).
   """
    mac_counters = []  # List to store MAC counters for each file
    
    # Read each CSV file and count occurrences of MAC addresses
    for csv_file in list_of_file_paths:
        mac_counter = Counter()
        
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) < 2:
                    continue  # Skip invalid rows
                mac_address = row[0].strip()
                mac_counter[mac_address] += 1
        
        mac_counters.append(mac_counter)
    
    # Find MAC addresses common to all files
    common_mac_ids = set(mac_counters[0].keys())
    for mac_counter in mac_counters[1:]:
        common_mac_ids.intersection_update(mac_counter.keys())
    
    if not common_mac_ids:
        print("No common MAC addresses found across all files.")
        return
    
   import csv
   from collections import Counter

   def count_common_mac_occurrences(list_of_file_paths, number_of_top_mac_ids, minimum_number_of_samples=None):
       mac_counters = []  # List to store MAC counters for each file
       
       # Read each CSV file and count occurrences of MAC addresses
       for csv_file in list_of_file_paths:
           mac_counter = Counter()
           
           with open(csv_file, 'r') as file:
               reader = csv.reader(file)
               for row in reader:
                   if len(row) < 2:
                       continue  # Skip invalid rows
                   mac_address = row[0].strip()
                   mac_counter[mac_address] += 1
           
           mac_counters.append(mac_counter)
       
       # Find MAC addresses common to all files
       common_mac_ids = set(mac_counters[0].keys())
       for mac_counter in mac_counters[1:]:
           common_mac_ids.intersection_update(mac_counter.keys())
       
       if not common_mac_ids:
           print("No common MAC addresses found across all files.")
           return
       
       # **Summing occurrences instead of taking the minimum**
       common_mac_counts = {mac: sum(counter[mac] for counter in mac_counters) for mac in common_mac_ids}
       
       # Sort in decreasing order of total count
       sorted_common_mac_counts = sorted(common_mac_counts.items(), key=lambda x: x[1], reverse=True)
       
       # Case 1: When minimum_number_of_samples is None
       if minimum_number_of_samples is None:
           top_mac_ids = sorted_common_mac_counts[:number_of_top_mac_ids]
           
           if len(common_mac_ids) < number_of_top_mac_ids:
               print("\nTotal number of common MAC IDs is less than the requested top count.")
           
           # Display the top common MAC addresses and their summed counts
           print(f"\nTop {number_of_top_mac_ids} Common MAC Addresses (Summed Count Across All Files):")
           for mac, count in top_mac_ids:
               print(f'{mac}: {count}')
           
           # Print MAC IDs in required list format
           print("\n[")
           for i, (mac, _) in enumerate(top_mac_ids):
               if i == len(top_mac_ids) - 1:
                   print(f'"{mac}"')  # Last entry without trailing comma
               else:
                   print(f'"{mac}", \\')
           print("]")
           
           # Find MAC ID with the least occurrences in the top list
           least_mac, least_count = min(top_mac_ids, key=lambda x: x[1], default=(None, None))
           print(f"\nMAC ID with the least occurrences in the top list: {least_mac} ({least_count} samples)")
       
       # Case 2: When minimum_number_of_samples is specified
       else:
           filtered_mac_ids = [(mac, count) for mac, count in sorted_common_mac_counts if count >= minimum_number_of_samples]
           
           if len(filtered_mac_ids) < number_of_top_mac_ids:
               print("\nThe required number_of_top_mac_ids with minimum_number_of_samples is not met.")
           else:
               print(f"\nMAC Addresses with at least {minimum_number_of_samples}:")
               for mac, count in filtered_mac_ids[:number_of_top_mac_ids]:
                   print(f'{mac}: {count}')
           
           # Print MAC IDs in required list format
           print("\n[")
           for i, (mac, _) in enumerate(filtered_mac_ids[:number_of_top_mac_ids]):
               if i == len(filtered_mac_ids[:number_of_top_mac_ids]) - 1:
                   print(f'"{mac}"')  # Last entry without trailing comma
               else:
                   print(f'"{mac}", \\')
           print("]")
           
           if filtered_mac_ids:
               least_mac, least_count = min(filtered_mac_ids, key=lambda x: x[1])
               print(f"\nMAC ID with the least occurrences in the filtered list: {least_mac} ({least_count} samples)")
           else:
               print("\nNo MAC IDs meet the minimum_number_of_samples requirement.")

   # Example Usage
   list_of_files = [
       "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_1.csv",
       "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_2.csv",
       "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_3.csv"
   ]
   number_of_top_mac_ids = 3
   minimum_samples = 5000  # Set an integer value if filtering by sample count is needed
   count_common_mac_occurrences(list_of_files, number_of_top_mac_ids, minimum_samples)
 # **Summing occurrences instead of taking the minimum**
    common_mac_counts = {mac: sum(counter[mac] for counter in mac_counters) for mac in common_mac_ids}
    
    # Sort in decreasing order of total count
    sorted_common_mac_counts = sorted(common_mac_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Case 1: When minimum_number_of_samples is None
    if minimum_number_of_samples is None:
        top_mac_ids = sorted_common_mac_counts[:number_of_top_mac_ids]
        
        if len(common_mac_ids) < number_of_top_mac_ids:
            print("\nTotal number of common MAC IDs is less than the requested top count.")
        
        # Display the top common MAC addresses and their summed counts
        print(f"\nTop {number_of_top_mac_ids} Common MAC Addresses (Summed Count Across All Files):")
        for mac, count in top_mac_ids:
            print(f'{mac}: {count}')
        
        # Print MAC IDs in required list format
        print("\n[")
        for i, (mac, _) in enumerate(top_mac_ids):
            if i == len(top_mac_ids) - 1:
                print(f'"{mac}"')  # Last entry without trailing comma
            else:
                print(f'"{mac}", \\')
        print("]")
        
        # Find MAC ID with the least occurrences in the top list
        least_mac, least_count = min(top_mac_ids, key=lambda x: x[1], default=(None, None))
        print(f"\nMAC ID with the least occurrences in the top list: {least_mac} ({least_count} samples)")
    
    # Case 2: When minimum_number_of_samples is specified
    else:
        filtered_mac_ids = [(mac, count) for mac, count in sorted_common_mac_counts if count >= minimum_number_of_samples]
        
        if len(filtered_mac_ids) < number_of_top_mac_ids:
            print("\nThe required number_of_top_mac_ids with minimum_number_of_samples is not met.")
        else:
            print(f"\nMAC Addresses with at least {minimum_number_of_samples}:")
            for mac, count in filtered_mac_ids[:number_of_top_mac_ids]:
                print(f'{mac}: {count}')
        
        # Print MAC IDs in required list format
        print("\n[")
        for i, (mac, _) in enumerate(filtered_mac_ids[:number_of_top_mac_ids]):
            if i == len(filtered_mac_ids[:number_of_top_mac_ids]) - 1:
                print(f'"{mac}"')  # Last entry without trailing comma
            else:
                print(f'"{mac}", \\')
        print("]")
        
        if filtered_mac_ids:
            least_mac, least_count = min(filtered_mac_ids, key=lambda x: x[1])
            print(f"\nMAC ID with the least occurrences in the filtered list: {least_mac} ({least_count} samples)")
        else:
            print("\nNo MAC IDs meet the minimum_number_of_samples requirement.")

# Example Usage
list_of_files = [
    "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_1.csv",
    "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_2.csv",
    "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_17_feb_3.csv"
]
number_of_top_mac_ids = 3
minimum_samples = 5000  # Set an integer value if filtering by sample count is needed
count_common_mac_occurrences(list_of_files, number_of_top_mac_ids, minimum_samples)





#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import csv

def low_pass_filter(signal, cutoff_freq=0.1):
    """
    Applies a low-pass filter using FFT.
    
    :param signal: 1D NumPy array (time-domain signal)
    :param cutoff_freq: Normalized cutoff frequency (0 to 1)
    :return: Filtered signal in the time domain
    """
    fft_coeffs = fft(signal)  # Compute FFT
    num_coeffs = len(fft_coeffs)
    
    # Zero out high frequencies beyond cutoff
    cutoff_index = int(cutoff_freq * num_coeffs)
    fft_coeffs[cutoff_index:-cutoff_index] = 0
    
    return np.real(ifft(fft_coeffs))  # Return real part after inverse FFT

def parse_csi_csv(file_path, target_macs, subcarriers, cutoff_freq=0.1, start_time=0, end_time=None):
    """
    Parses the CSI CSV file, filters data for specific MAC addresses,
    extracts amplitude values for selected subcarriers, applies FFT-based filtering,
    and plots amplitude vs. time for a given range.

    :param file_path: Path to the CSV file
    :param target_macs: List of MAC addresses to filter
    :param subcarriers: List of subcarrier indices (1 to 64)
    :param cutoff_freq: Low-pass filter cutoff frequency
    :param start_time: Start index for plotting
    :param end_time: End index for plotting (None means full range)
    """
    
    data = {mac: [] for mac in target_macs}  # Store data separately for each MAC
    timestamps = {mac: [] for mac in target_macs}

    # Read CSV file and extract CSI data
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            if len(line) < 2:
                continue  # Skip invalid rows
            
            mac = line[0].strip()
            if mac in target_macs:
                csi_values = list(map(int, line[1].split()))
                
                amplitudes = np.array(csi_values[::2])  # Extract magnitudes
                
                selected_amplitudes = amplitudes[np.array(subcarriers) - 1]  # Convert 1-based to 0-based
                
                data[mac].append(selected_amplitudes)
                timestamps[mac].append(len(data[mac]))  # Use row index as time for simplicity

    # Process and plot data for each MAC
    for mac in target_macs:
        if not data[mac]:
            print(f"No data found for MAC: {mac}")
            continue
        
        raw_data = np.array(data[mac])  # Convert list to NumPy array
        filtered_data = np.apply_along_axis(low_pass_filter, axis=0, arr=raw_data, cutoff_freq=cutoff_freq)
        
        # Define time range
        if end_time is None:
            end_time = len(filtered_data)
        if start_time < 0 or end_time > len(filtered_data) or start_time >= end_time:
            print(f"Invalid time range for {mac}: {start_time} to {end_time}")
            continue

        # Plot raw and filtered data
        plt.figure(figsize=(12, 6))
        
        for i, subcarrier in enumerate(subcarriers):
            plt.plot(timestamps[mac][start_time:end_time], raw_data[start_time:end_time, i], label=f'Raw Subcarrier {subcarrier}', linestyle='dashed', alpha=0.5)
            plt.plot(timestamps[mac][start_time:end_time], filtered_data[start_time:end_time, i], label=f'Filtered Subcarrier {subcarrier}')

        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.title(f"CSI Amplitude vs. Time for MAC {mac} ({start_time} to {end_time})")
        plt.legend()
        plt.grid(True)
        plt.show()

# Example Usage
file_path = "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_5aps_fixedpos_16_feb_3.csv"
target_macs = ["8C:4F:00:3C:BF:4D", "34:5F:45:A9:A4:19"]  # List of MAC addresses
selected_subcarriers = [50]  # Select subcarriers of interest
start_sample = 400
end_sample = 800

parse_csi_csv(file_path, target_macs, selected_subcarriers, cutoff_freq=0.1, start_time=start_sample, end_time=end_sample)


#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import csv

def low_pass_filter(signal, cutoff_freq=0.1):
    """
    Applies a low-pass filter using FFT.
    
    :param signal: 1D NumPy array (time-domain signal)
    :param cutoff_freq: Normalized cutoff frequency (0 to 1)
    :return: Filtered signal in the time domain
    """
    fft_coeffs = fft(signal)  # Compute FFT
    num_coeffs = len(fft_coeffs)
    
    # Zero out high frequencies beyond cutoff
    cutoff_index = int(cutoff_freq * num_coeffs)
    fft_coeffs[cutoff_index:-cutoff_index] = 0
    
    return np.real(ifft(fft_coeffs))  # Return real part after inverse FFT

def parse_csi_csv(file_path, target_macs, subcarriers, cutoff_freq=0.1, start_time=0, end_time=None, plot_raw =True, plot_fil = True):
    """
    Parses the CSI CSV file, filters data for specific MAC addresses,
    extracts amplitude values for selected subcarriers, applies FFT-based filtering,
    and plots all MACs on the same figure.

    :param file_path: Path to the CSV file
    :param target_macs: List of MAC addresses to filter
    :param subcarriers: List of subcarrier indices (1 to 64)
    :param cutoff_freq: Low-pass filter cutoff frequency
    :param start_time: Start index for plotting
    :param end_time: End index for plotting (None means full range)
    """
    
    data = {mac: [] for mac in target_macs}  # Store data separately for each MAC
    timestamps = {mac: [] for mac in target_macs}

    # Read CSV file and extract CSI data
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            if len(line) < 2:
                continue  # Skip invalid rows
            
            mac = line[0].strip()
            if mac in target_macs:
                csi_values = list(map(int, line[1].split()))
                
                amplitudes = np.array(csi_values[::2])  # Extract magnitudes
                
                selected_amplitudes = amplitudes[np.array(subcarriers) - 1]  # Convert 1-based to 0-based
                
                data[mac].append(selected_amplitudes)
                timestamps[mac].append(len(data[mac]))  # Use row index as time for simplicity

    # Create a single figure for all MACs
    plt.figure(figsize=(12, 6))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Predefined colors for differentiation
    mac_colors = {mac: colors[i % len(colors)] for i, mac in enumerate(target_macs)}  # Assign colors to MACs

    # Process and plot data for each MAC
    for mac in target_macs:
        if not data[mac]:
            print(f"No data found for MAC: {mac}")
            continue
        
        raw_data = np.array(data[mac])  # Convert list to NumPy array
        filtered_data = np.apply_along_axis(low_pass_filter, axis=0, arr=raw_data, cutoff_freq=cutoff_freq)
        
        # Define time range
        if end_time is None:
            end_time = len(filtered_data)
        if start_time < 0 or end_time > len(filtered_data) or start_time >= end_time:
            print(f"Invalid time range for {mac}: {start_time} to {end_time}")
            continue

        # Plot raw and filtered data for each subcarrier
        
        for i, subcarrier in enumerate(subcarriers):
            if plot_raw:
                plt.plot(
                    timestamps[mac][start_time:end_time], 
                    raw_data[start_time:end_time, i], 
                    linestyle='dashed', alpha=0.5, color=mac_colors[mac], 
                    label=f'Raw {mac} SC {subcarrier}' if i == 0 else "_nolegend_"
                    )
            
            if plot_fil:
                plt.plot(
                    timestamps[mac][start_time:end_time], 
                    filtered_data[start_time:end_time, i], 
                    linestyle='solid', color=mac_colors[mac], 
                    label=f'Filtered {mac} SC {subcarrier}' if i == 0 else "_nolegend_"
                )

    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title(f"CSI Amplitude vs. Time ({start_time} to {end_time})")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example Usage
file_path = "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_5aps_fixedpos_16_feb_3.csv"
target_macs = ["8C:4F:00:3C:BF:4D", "34:5F:45:A9:A4:19"]  # List of MAC addresses
selected_subcarriers = [50, 40]  # Select subcarriers of interest
start_sample = 1000
end_sample = 1100

parse_csi_csv(file_path=file_path, target_macs = target_macs, subcarriers = selected_subcarriers, cutoff_freq=0.1, start_time=start_sample, end_time=end_sample, plot_raw =False, plot_fil = True)



#%%


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import csv

def low_pass_filter(signal, cutoff_freq=0.1):
    """
    Applies a low-pass filter using FFT.
    
    :param signal: 1D NumPy array (time-domain signal)
    :param cutoff_freq: Normalized cutoff frequency (0 to 1)
    :return: Filtered signal in the time domain
    """
    fft_coeffs = fft(signal)  # Compute FFT
    num_coeffs = len(fft_coeffs)
    
    # Zero out high frequencies beyond cutoff
    cutoff_index = int(cutoff_freq * num_coeffs)
    fft_coeffs[cutoff_index:-cutoff_index] = 0
    
    return np.real(ifft(fft_coeffs))  # Return real part after inverse FFT

def parse_csi_csv(file_path, target_macs, subcarriers, cutoff_freq=0.1, start_time=0, end_time=None, plot_raw=True, plot_fil=True):
    """
    Parses the CSI CSV file, filters data for specific MAC addresses,
    extracts amplitude values for selected subcarriers, applies FFT-based filtering,
    and plots all MACs on the same figure with unique colors.

    :param file_path: Path to the CSV file
    :param target_macs: List of MAC addresses to filter
    :param subcarriers: List of subcarrier indices (1 to 64)
    :param cutoff_freq: Low-pass filter cutoff frequency
    :param start_time: Start index for plotting
    :param end_time: End index for plotting (None means full range)
    :param plot_raw: Boolean to plot raw signal
    :param plot_fil: Boolean to plot filtered signal
    """
    
    data = {mac: [] for mac in target_macs}  # Store data separately for each MAC
    timestamps = {mac: [] for mac in target_macs}

    # Read CSV file and extract CSI data
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            if len(line) < 2:
                continue  # Skip invalid rows
            
            mac = line[0].strip()
            if mac in target_macs:
                csi_values = list(map(int, line[1].split()))
                
                amplitudes = np.array(csi_values[::2])  # Extract magnitudes
                
                selected_amplitudes = amplitudes[np.array(subcarriers) - 1]  # Convert 1-based to 0-based
                
                data[mac].append(selected_amplitudes)
                timestamps[mac].append(len(data[mac]))  # Use row index as time for simplicity

    # Create a single figure for all MACs
    plt.figure(figsize=(12, 6))

    # Generate unique colors for each MAC-subcarrier pair
    color_map = plt.cm.get_cmap("tab10", len(target_macs) * len(subcarriers))
    color_idx = 0  # Track color index

    # Process and plot data for each MAC
    for mac in target_macs:
        if not data[mac]:
            print(f"No data found for MAC: {mac}")
            continue
        
        raw_data = np.array(data[mac])  # Convert list to NumPy array
        filtered_data = np.apply_along_axis(low_pass_filter, axis=0, arr=raw_data, cutoff_freq=cutoff_freq)
        
        # Define time range
        if end_time is None:
            end_time = len(filtered_data)
        if start_time < 0 or end_time > len(filtered_data) or start_time >= end_time:
            print(f"Invalid time range for {mac}: {start_time} to {end_time}")
            continue

        # Plot raw and filtered data for each subcarrier
        for i, subcarrier in enumerate(subcarriers):
            unique_color = color_map(color_idx)  # Assign a unique color
            color_idx += 1

            if plot_raw:
                plt.plot(
                    timestamps[mac][start_time:end_time], 
                    raw_data[start_time:end_time, i], 
                    linestyle='dashed', alpha=0.5, color=unique_color, 
                    label=f'Raw {mac} SC {subcarrier}'
                )
            
            if plot_fil:
                plt.plot(
                    timestamps[mac][start_time:end_time], 
                    filtered_data[start_time:end_time, i], 
                    linestyle='solid', color=unique_color, 
                    label=f'Filtered {mac} SC {subcarrier}'
                )

    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title(f"CSI Amplitude vs. Time ({start_time} to {end_time})")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example Usage
file_path = "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_5aps_fixedpos_16_feb_3.csv"
target_macs = ["8C:4F:00:3C:BF:4D"]  # List of MAC addresses
#target_macs = ["8C:4F:00:3C:BF:4D", "34:5F:45:A9:A4:19"]  # List of MAC addresses
selected_subcarriers = [50]  # Select subcarriers of interest
start_sample = 1000
end_sample = 1100

parse_csi_csv(
    file_path=file_path,
    target_macs=target_macs,
    subcarriers=selected_subcarriers,
    cutoff_freq=0.1,
    start_time=start_sample,
    end_time=end_sample,
    plot_raw=True,
    plot_fil=False
)



#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import csv

def low_pass_filter(signal, cutoff_freq=0.1):
    """
    Applies a low-pass filter using FFT.
    
    :param signal: 1D NumPy array (time-domain signal)
    :param cutoff_freq: Normalized cutoff frequency (0 to 1)
    :return: Filtered signal in the time domain
    """
    fft_coeffs = fft(signal)  # Compute FFT
    num_coeffs = len(fft_coeffs)
    
    # Zero out high frequencies beyond cutoff
    cutoff_index = int(cutoff_freq * num_coeffs)
    fft_coeffs[cutoff_index:-cutoff_index] = 0
    
    return np.real(ifft(fft_coeffs))  # Return real part after inverse FFT

def parse_csi_csv(file_path, target_macs, subcarriers, cutoff_freq=0.1, start_time=0, end_time=None, plot_raw=True, plot_fil=True):
    """
    Parses the CSI CSV file, filters data for specific MAC addresses,
    extracts amplitude values for selected subcarriers, applies FFT-based filtering,
    and plots all MACs on the same figure with unique colors.

    :param file_path: Path to the CSV file
    :param target_macs: List of MAC addresses to filter
    :param subcarriers: List of subcarrier indices (1 to 64)
    :param cutoff_freq: Low-pass filter cutoff frequency
    :param start_time: Start index for plotting
    :param end_time: End index for plotting (None means full range)
    :param plot_raw: Boolean to plot raw signal
    :param plot_fil: Boolean to plot filtered signal
    """
    
    data = {mac: [] for mac in target_macs}  # Store data separately for each MAC
    timestamps = {mac: [] for mac in target_macs}

    # Read CSV file and extract CSI data
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            if len(line) < 2:
                continue  # Skip invalid rows
            
            mac = line[0].strip()
            if mac in target_macs:
                csi_values = list(map(int, line[1].split()))
                
                # Extract real and imaginary parts
                imag_parts = np.array(csi_values[0::2])  # Odd indices -> Imag
                real_parts = np.array(csi_values[1::2])  # Even indices -> Real
                
                # Compute amplitude
                amplitudes = np.sqrt(real_parts**2 + imag_parts**2)  
                
                # Select subcarriers (1-based index)
                selected_amplitudes = amplitudes[np.array(subcarriers) - 1]
                
                data[mac].append(selected_amplitudes)
                timestamps[mac].append(len(data[mac]))  # Use row index as time for simplicity

    # Create a single figure for all MACs
    plt.figure(figsize=(12, 6))

    # Generate unique colors for each MAC-subcarrier pair
    color_map = plt.cm.get_cmap("tab10", len(target_macs) * len(subcarriers))
    color_idx = 0  # Track color index

    # Process and plot data for each MAC
    for mac in target_macs:
        if not data[mac]:
            print(f"No data found for MAC: {mac}")
            continue
        
        raw_data = np.array(data[mac])  # Convert list to NumPy array
        filtered_data = np.apply_along_axis(low_pass_filter, axis=0, arr=raw_data, cutoff_freq=cutoff_freq)
        
        # Define time range
        if end_time is None:
            end_time = len(filtered_data)
        if start_time < 0 or end_time > len(filtered_data) or start_time >= end_time:
            print(f"Invalid time range for {mac}: {start_time} to {end_time}")
            continue

        # Plot raw and filtered data for each subcarrier
        for i, subcarrier in enumerate(subcarriers):
            unique_color = color_map(color_idx)  # Assign a unique color
            color_idx += 1

            if plot_raw:
                plt.plot(
                    timestamps[mac][start_time:end_time], 
                    raw_data[start_time:end_time, i], 
                    linestyle='solid', alpha=0.5, color=unique_color, 
                    label=f'Raw {mac} SC {subcarrier}'
                )
            
            if plot_fil:
                plt.plot(
                    timestamps[mac][start_time:end_time], 
                    filtered_data[start_time:end_time, i], 
                    linestyle='dashed', color=unique_color, 
                    label=f'Filtered {mac} SC {subcarrier}'
                )

    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title(f"CSI Amplitude vs. Time ({start_time} to {end_time})")
    plt.legend()
    plt.grid(True)
    plt.show()

file_path = "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_5aps_fixedpos_16_feb_3.csv"
target_macs = ["34:5F:45:A9:A4:19"]  # List of MAC addresses
#target_macs = ["34:5F:45:A9:A4:19" , "34:5F:45:A8:3C:19"]  # List of MAC addresses
'''
target_macs = ["00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"00:FC:BA:38:4B:02"]
 '''   
#target_macs = ["00:FC:BA:38:4B:00"]    
    
selected_subcarriers = [40,41]  # Select subcarriers of interest
start_sample = 1000
end_sample = 1100

parse_csi_csv(
    file_path=file_path,
    target_macs=target_macs,
    subcarriers=selected_subcarriers,
    cutoff_freq=.1,
    start_time=start_sample,
    end_time=end_sample,
    plot_raw=False,
    plot_fil=True
)





#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import csv

def low_pass_filter(signal, cutoff_freq=0.1):
    """
    Applies a low-pass filter using FFT.
    
    :param signal: 1D NumPy array (time-domain signal)
    :param cutoff_freq: Normalized cutoff frequency (0 to 1)
    :return: Filtered signal in the time domain
    """
    fft_coeffs = fft(signal)  # Compute FFT
    num_coeffs = len(fft_coeffs)
    
    # Zero out high frequencies beyond cutoff
    cutoff_index = int(cutoff_freq * num_coeffs)
    fft_coeffs[cutoff_index:-cutoff_index] = 0
    
    return np.real(ifft(fft_coeffs))  # Return real part after inverse FFT

def parse_csi_csv(file_path, target_macs, subcarriers, cutoff_freq=0.1, start_time=0, end_time=None, plot_raw=True, plot_fil=True, plot_phase=True):
    """
    Parses the CSI CSV file, filters data for specific MAC addresses,
    extracts amplitude and phase values for selected subcarriers,
    applies FFT-based filtering, and plots the results.

    :param file_path: Path to the CSV file
    :param target_macs: List of MAC addresses to filter
    :param subcarriers: List of subcarrier indices (1 to 64)
    :param cutoff_freq: Low-pass filter cutoff frequency
    :param start_time: Start index for plotting
    :param end_time: End index for plotting (None means full range)
    :param plot_raw: Boolean to plot raw amplitude
    :param plot_fil: Boolean to plot filtered amplitude
    :param plot_phase: Boolean to plot filtered phase
    """
    
    data_amp = {mac: [] for mac in target_macs}  # Amplitude data
    data_phase = {mac: [] for mac in target_macs}  # Phase data
    timestamps = {mac: [] for mac in target_macs}

    # Read CSV file and extract CSI data
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            if len(line) < 2:
                continue  # Skip invalid rows
            
            mac = line[0].strip()
            if mac in target_macs:
                csi_values = list(map(int, line[1].split()))
                
                # Extract real and imaginary parts
                imag_parts = np.array(csi_values[0::2])  # Odd indices -> Imag
                real_parts = np.array(csi_values[1::2])  # Even indices -> Real
                
                # Compute amplitude and phase
                amplitudes = np.sqrt(real_parts**2 + imag_parts**2)  
                phases = np.arctan2(imag_parts, real_parts)  # Compute phase
                
                # Select subcarriers (1-based index)
                selected_amplitudes = amplitudes[np.array(subcarriers) - 1]
                selected_phases = phases[np.array(subcarriers) - 1]
                
                data_amp[mac].append(selected_amplitudes)
                data_phase[mac].append(selected_phases)
                timestamps[mac].append(len(data_amp[mac]))  # Use row index as time for simplicity

    # === Plot Amplitude ===
    plt.figure(figsize=(12, 6))
    color_map = plt.cm.get_cmap("tab10", len(target_macs) * len(subcarriers))
    color_idx = 0

    for mac in target_macs:
        if not data_amp[mac]:
            print(f"No data found for MAC: {mac}")
            continue
        
        raw_data = np.array(data_amp[mac])
        filtered_data = np.apply_along_axis(low_pass_filter, axis=0, arr=raw_data, cutoff_freq=cutoff_freq)
        
        if end_time is None:
            end_time = len(filtered_data)

        if start_time < 0 or end_time > len(filtered_data) or start_time >= end_time:
            print(f"Invalid time range for {mac}: {start_time} to {end_time}")
            continue

        for i, subcarrier in enumerate(subcarriers):
            unique_color = color_map(color_idx)
            color_idx += 1

            if plot_raw:
                plt.plot(
                    timestamps[mac][start_time:end_time], 
                    raw_data[start_time:end_time, i], 
                    linestyle='solid', alpha=0.5, color=unique_color, 
                    label=f'Raw {mac} SC {subcarrier}'
                )
            
            if plot_fil:
                plt.plot(
                    timestamps[mac][start_time:end_time], 
                    filtered_data[start_time:end_time, i], 
                    linestyle='dashed', color=unique_color, 
                    label=f'Filtered {mac} SC {subcarrier}'
                )

    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title(f"CSI Amplitude vs. Time ({start_time} to {end_time})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # === Plot Phase ===
    if plot_phase:
        plt.figure(figsize=(12, 6))
        color_idx = 0  # Reset color index

        for mac in target_macs:
            if not data_phase[mac]:
                continue
            
            raw_phase = np.array(data_phase[mac])
            filtered_phase = np.apply_along_axis(low_pass_filter, axis=0, arr=raw_phase, cutoff_freq=cutoff_freq)

            for i, subcarrier in enumerate(subcarriers):
                unique_color = color_map(color_idx)
                color_idx += 1

                plt.plot(
                    timestamps[mac][start_time:end_time], 
                    filtered_phase[start_time:end_time, i], 
                    linestyle='dashed', color=unique_color, 
                    label=f'Filtered Phase {mac} SC {subcarrier}'
                )

        plt.xlabel("Time (samples)")
        plt.ylabel("Phase (radians)")
        plt.title(f"CSI Phase vs. Time ({start_time} to {end_time})")
        plt.legend()
        plt.grid(True)
        plt.show()

file_path = "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_5aps_fixedpos_16_feb_3.csv"
#target_macs = ["34:5F:45:A9:A4:19"]
target_macs = ["34:5F:45:A9:A4:19" , "34:5F:45:A8:3C:19", "20:43:A8:64:3A:C1"]  # List of MAC addresses
selected_subcarriers = [40]
start_sample = 1000
end_sample = 1100

parse_csi_csv(
    file_path=file_path,
    target_macs=target_macs,
    subcarriers=selected_subcarriers,
    cutoff_freq=0.1,
    start_time=start_sample,
    end_time=end_sample,
    plot_raw=True,
    plot_fil=False,
    plot_phase=True
)










