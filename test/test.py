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




#%%

import os

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

# Example usage: Directly calling the function
folder_path = "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/esp_printed_antenna/04_feb_2025"
n = 5  # Number of files you want to retrieve

get_n_csv_filepaths(folder_path, n)

#%% esp32 systematic data experiments

#IMPORTS
from count_common_mac import count_common_mac_occurrences
from amp_phase_fft_plot import parse_csi_amp_phase_fft_plot
from csv_merge import combine_csv_files
from mac_id_counter import count_mac_occurrences
from get_n_csv_filepaths import get_n_csv_filepaths

from csi_dataset_creator_fixed_id import process_csv_fixed_id
from csi_dataset_creator_fixed_id_uniform_sampling import process_csv_fixed_id_uniform_sampling
from csi_dataset_creator import process_csv
from mean_norm import mean_norm
from train_test import train, test
from train_test_loader import train_test_loader
from models import SimpleCNN
import torch.nn as nn
import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#%% 

list_of_file_paths = [ \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/24_feb_25_p3_04_31_05_15.csv", \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/24_feb_25_p4_05_17_06_55.csv", \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/24_feb_25_p5_07_00_09_00.csv", \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/25_feb_25_p6_1_00_01_15.csv", \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/25_feb_25_p7_01_15_03_00.csv", \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/25_feb_25_p4_03_05_05_00.csv", \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/25_feb_25_p5_05_05_07_00.csv", \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/25_feb_25_p6_07_05_09_00.csv", \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/26_feb_25_p3_01_00_01_50.csv", \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/26_feb_25_p4_05_00_07_00.csv", \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/26_feb_25_p5_07_00_.csv", \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/27_feb_25_p4_11_50_01_05.csv", \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/27_feb_25_p5_01_10_01_45.csv", \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/27_feb_25_p5_02_50_.csv", \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/27_feb_25_p6_03_15_05_00.csv", \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/27_feb_25_p7_05_00_07_00.csv", \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/27_feb_25_p8_07_10_09_00.csv", \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/28_feb_25_p7_04_40_.csv", \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/28_feb_25_p8_05_15_07_00.csv", \
    "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection/28_feb_25_p9_07_00_9_00.csv" \
]

count_common_mac_occurrences(list_of_file_paths, number_of_top_mac_ids=15, minimum_number_of_samples=None)

output_file =  "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/test/systemtic_first_20_merged/merged.csv"   
combine_csv_files(file_paths= list_of_file_paths, output_file=output_file)


#%% Training and Testing 

# Prompt the user for the file path
file_path = input("Please enter the file path to the CSV file: ")

# Process the CSV file
data, labels = process_csv_fixed_id_uniform_sampling(file_path = file_path , mac_id_list = \
[
"6C:B2:AE:39:1A:A0", \
"6C:B2:AE:39:1A:A1", \
"6C:B2:AE:39:1A:A2", \
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"00:FC:BA:38:4B:02", \
"FE:19:28:38:54:40", \
"70:0F:6A:DE:EC:A0", \
"70:0F:6A:DE:EC:A1", \
"70:0F:6A:DE:EC:A2", \
"C8:28:E5:44:3B:00", \
"00:FC:BA:27:63:00", \
"00:FC:BA:27:63:01", \
"00:FC:BA:27:63:02", \
"70:0F:6A:FC:51:81" \
], max_samples_per_mac=18000)
data = data.unsqueeze(1)
dataset = mean_norm(data)
train_loader, test_loader = train_test_loader(dataset, labels)

num_classes = 15
learning_rate = 0.001
num_epochs = 50

# Model setup
model = SimpleCNN(num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Model Training
model.train()
train(model=model, train_loader=train_loader, test_loader=test_loader, criterion=criterion, optimizer=optimizer, num_epochs=num_epochs)

# Model Testing
model.eval()
_ = test(model, test_loader)



#%% Testing on new test dataset

file_path = input("Please enter the file path to the test CSV file: ")
data, labels = process_csv_fixed_id_uniform_sampling(file_path = file_path , mac_id_list = [ \
"6C:B2:AE:39:1A:A0", \
"6C:B2:AE:39:1A:A1", \
"6C:B2:AE:39:1A:A2", \
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"00:FC:BA:38:4B:02", \
"FE:19:28:38:54:40", \
"70:0F:6A:DE:EC:A0", \
"70:0F:6A:DE:EC:A1", \
"70:0F:6A:DE:EC:A2", \
"C8:28:E5:44:3B:00", \
"00:FC:BA:27:63:00", \
"00:FC:BA:27:63:01", \
"00:FC:BA:27:63:02", \
"70:0F:6A:FC:51:81" \
], max_samples_per_mac=5000)
data = data.unsqueeze(1)
dataset = mean_norm(data)
train_loader, test_loader = train_test_loader(dataset, labels)

model.eval()
_ = test(model, train_loader)


#%%

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

# Example usage:
folder_path = "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/csi_rff_data/systematic_collection"  # Change this to your actual folder path
n = 20 # Change this to the desired number of files

get_n_csv_filepaths(folder_path, n)


#%%


#imports
import torch.nn as nn
import torch
from torchvision.models import ResNet50_Weights, VGG16_Weights, Inception_V3_Weights
from torchvision import models
from transformers import ViTConfig, ViTForImageClassification, AdamW
#from utilities import get_positional_encoding
import math


# vit_model_2

'''
This model is used to do classification task on caa input data of shape 1000by8
using ViT based model. We are using a built-in transformer model from huggingface
called ViTForImageClassification which takes in a ViTConfig file which contains the
details of the model like input size, number of classes, attention head etc. We 
wrap this inside the nn.module() to create the vit_model_2. In this model we configure
the ViTConfig to do tokenziation by taking each of the 1000by1 columns in the whole
1000by8 and embedding them. This is achieved by setting the convolutional filter
patch size as (1000,1).
'''

class vit_model_2(nn.Module):  # Defining a custom ViT model class inheriting from nn.Module
    def __init__(self, input_dim=(64, 2), num_classes=15, hidden_size=768, 
                 num_attention_heads=12, num_hidden_layers=12, intermediate_size=3072, 
                 patch_size=(64, 1), num_channels=1):
        
        """
        Initializes the Vision Transformer (ViT) model with custom configurations.

        Args:
        - input_dim (tuple): Dimensions of the input data (height, width). Default is (1000, 8).
        - num_classes (int): Number of output classes for classification. Default is 300.
        - hidden_size (int): Size of the transformer hidden layers. Default is 768.
        - num_attention_heads (int): Number of attention heads in the transformer layers. Default is 12.
        - num_hidden_layers (int): Number of transformer layers. Default is 12.
        - intermediate_size (int): Size of the intermediate feed-forward layer in the transformer. Default is 3072.
        - patch_size (tuple): Size of each patch the model processes. Default is (1000, 1).
        - num_channels (int): Number of input channels. Default is 1 for grayscale data.
        """
        
        super(vit_model_2, self).__init__()  # Calls the constructor of the parent class (nn.Module)
        
        # Store the model hyperparameters
        self.input_dim = input_dim  # Input image dimensions (Height, Width)
        self.num_classes = num_classes  # Number of classification labels
        self.hidden_size = hidden_size  # Transformer hidden layer size
        self.num_attention_heads = num_attention_heads  # Number of attention heads
        self.num_hidden_layers = num_hidden_layers  # Number of transformer layers
        self.intermediate_size = intermediate_size  # Feed-forward network size
        self.patch_size = patch_size  # Patch size for dividing the input image
        self.num_channels = num_channels  # Number of channels (e.g., grayscale = 1, RGB = 3)

        # Create ViT Configuration object with the specified parameters
        self.ViTConfig = ViTConfig(
            image_size=self.input_dim,  # Specifies the input image dimensions (height, width)
            num_labels=self.num_classes,  # Number of classes in the output classification
            hidden_size=self.hidden_size,  # Size of hidden layers in the transformer
            num_attention_heads=self.num_attention_heads,  # Number of self-attention heads per transformer layer
            num_hidden_layers=self.num_hidden_layers,  # Total transformer encoder layers
            intermediate_size=self.intermediate_size,  # Size of the feed-forward layer inside each transformer block
            patch_size=self.patch_size,  # Size of image patches that will be fed to the transformer
            num_channels=self.num_channels,  # Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        )

        # Initialize the Vision Transformer model for image classification using the defined configuration
        self.ViTForImageClassification = ViTForImageClassification(self.ViTConfig)
        
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
        - x (torch.Tensor): Input tensor representing an image or batch of images.

        Returns:
        - torch.Tensor: The output logits from the ViT classification model.
        """
        x = self.ViTForImageClassification(x)  # Pass input through the ViT model
        return x

###############################################################################


#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:31:10 2025

@author: fawaz
"""

import csv
import torch
import random

def parse_csi_data(csi_row):
    """
    Parses a single row of CSI data into a 64x2 PyTorch tensor.
    """
    csi_values = csi_row.split()
    if len(csi_values) != 128:
        return None  # Skip invalid CSI rows
    csi_tensor = []
    for i in range(0, 128, 2):
        try:
            magnitude = float(csi_values[i])
            angle = float(csi_values[i + 1])
            csi_tensor.append([magnitude, angle])
        except ValueError:
            return None  # Skip rows with invalid numeric values
    return torch.tensor(csi_tensor)

def process_csv_fixed_id_uniform_sampling(file_path, mac_id_list, max_samples_per_mac=50000):
    """
    Processes a CSV file to extract CSI data for specific MAC addresses and assigns labels based on their order in mac_id_list.
    Instead of selecting the first max_samples_per_mac entries, this function selects uniformly from all available entries.
    """
    mac_entries = {mac: [] for mac in mac_id_list}  # Store all CSI data for each MAC
    
    # Read CSV file and collect all valid CSI entries for each MAC
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) != 2:
                continue  # Skip invalid rows
            current_mac_id, csi_row = row  
            if current_mac_id not in mac_id_list:
                continue  # Skip MACs not in the specified list
            
            csi_tensor = parse_csi_data(csi_row)
            if csi_tensor is not None:
                mac_entries[current_mac_id].append(csi_tensor)  # Store valid CSI tensor
    
    # Randomly select up to max_samples_per_mac for each MAC
    data = []
    labels = []
    mac_id_to_label = {mac: i for i, mac in enumerate(mac_id_list)}  # Assign labels based on order

    for mac, entries in mac_entries.items():
        sample_size = min(len(entries), max_samples_per_mac)
        sampled_entries = random.sample(entries, sample_size)  # Uniform random selection

        data.extend(sampled_entries)
        labels.extend([mac_id_to_label[mac]] * sample_size)

    if data:
        data_ = torch.stack(data)
        labels_ = torch.tensor(labels, dtype=torch.long)
        return data_, labels_
    else:
        return None, None  # Return None if no valid data was processed


#%% imports

# 1) imports
import models
import torch
import torch.nn as nn
import torch.optim as optim
from utilities import load_data_from_csv, train_test_loader, train, test, train_inception, \
    mean_norm,copy_columns, train_vit, load_data_from_csv_vit_model_1, get_positional_encoding, \
    calculate_test_accuracy_vit_model_1, train_vit_model_1, CustomDataset_vit_model_1, \
        CustomDataset_vit_model_2, train_vit_model_2, test_vit_model_2
from models import SimpleCNN, CustomResNet50, CustomVgg16, CustomInceptionV3, vit_model_1, vit_model_2
from torch.utils.data import Subset, DataLoader, TensorDataset, random_split
from transformers import AdamW

print("Done imports")

#%%
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from datetime import datetime
from tqdm import tqdm
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

# 16) CustomDataset_vit_model_1

# Define a custom dataset class inheriting from PyTorch's Dataset class
class CustomDataset_vit_model_1(Dataset):
    def __init__(self, data, labels):
        """
        Initialize the dataset.
        
        Args:
            data (torch.Tensor): 
                - The input data tensor of shape (N, 1000, 8)
                  where:
                    - N = Number of samples
                    - 1000 = Number of time steps (or features) per sample
                    - 8 = Number of channels or dimensions per time step

            labels (torch.Tensor): 
                - Tensor of labels of shape (N,)
                  where:
                    - N = Number of samples
                    - Each label is a single integer representing the class index
        
        Example:
            data.shape = (440, 1000, 8)  # 440 samples, 1000 time steps, 8 channels
            labels.shape = (440,)         # 440 labels (one per sample)
        """
        # Store the data and labels as class attributes
        self.data = data
        self.labels = labels
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset (length of the data tensor)
        
        Example:
            If data.shape = (440, 1000, 8), this will return 440
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample and its corresponding label.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            torch.Tensor: 
                - Data sample of shape (1, 1000, 8)  
                - The unsqueeze operation adds a new dimension at the beginning 
                  to represent the "channel" dimension, which is often required 
                  when working with convolutional or transformer models.
              
            torch.Tensor:
                - Corresponding label as a single integer.
        
        Example:
            If data[idx] has shape (1000, 8), after unsqueeze:
            sample.shape = (1, 1000, 8)
        
        Notes:
            - `unsqueeze(0)` converts a 2D tensor (1000, 8) into a 3D tensor (1, 1000, 8).
            - This is useful if the model expects a channel dimension in the input.
        """
        sample = self.data[idx].unsqueeze(0)  # (1000, 8) -> (1, 1000, 8)
        label = self.labels[idx]  # Integer label
        return sample, label
    


# 17) CustomDataset_vit_model_2

class CustomDataset_vit_model_2(Dataset):
    def __init__(self, data, labels):
        """
        Initialize the CustomDataset class.
        
        Args:
            data (torch.Tensor): A tensor containing the input data of shape (N, 1000, 8),
                                 where:
                                 - N = number of samples
                                 - 1000 = sequence length (or time steps)
                                 - 8 = number of features (or channels)
            labels (torch.Tensor): A tensor containing the class labels of shape (N,),
                                   where N = number of samples.
        
        Example:
            data shape: (10000, 1000, 8)
            labels shape: (10000,)
        """
        self.data = data      # Store the input data
        self.labels = labels  # Store the labels
    
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        This allows the DataLoader to know how many samples are available.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)  # Return the length of the dataset
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample and its corresponding label based on the given index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            torch.Tensor: A data sample of shape (1, 1000, 8), where:
                          - 1 = channel dimension (for compatibility with CNN/Transformer models)
                          - 1000 = sequence length
                          - 8 = number of features (or channels)
            torch.Tensor: Corresponding label (scalar value).
        
        Example:
            If the original sample shape is (1000, 8), it is reshaped to (1, 1000, 8)
            using `unsqueeze(0)`, which adds a channel dimension.
        """
        # Extract the sample at the specified index
        sample = self.data[idx]  # Shape: (1000, 8)
        
        # Add a channel dimension at the beginning to make it compatible with CNN/Transformer models
        sample = sample.unsqueeze(0)  # Shape becomes: (1, 1000, 8)
        
        # Extract the corresponding label
        label = self.labels[idx]
        
        return sample, label
    
    
#%%

# 18) train_vit_model_2

def train_vit_model_2(model, train_loader, optimizer, loss_fn, device):
    """
    Function to train a model for one epoch.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader providing batches of training data.
        optimizer (torch.optim.Optimizer): Optimizer to update model weights (e.g., Adam, SGD).
        loss_fn (torch.nn.Module): Loss function to compute the error (e.g., CrossEntropyLoss).
        device (torch.device): Device to run the training on ('cuda' or 'cpu').

    Returns:
        tuple: (avg_loss, avg_accuracy)
            - avg_loss (float): Average loss over the training dataset.
            - avg_accuracy (float): Average accuracy over the training dataset.
    """

    # Set the model to training mode
    # This enables certain layers like dropout and batch normalization to behave differently during training.
    model.train()

    # Initialize variables to track the total loss, correct predictions, and total samples
    total_loss = 0
    total_correct = 0
    total_samples = 0

    # Loop over each batch of data provided by the train_loader
    for images, labels in train_loader:
        # Move input data and labels to the specified device (GPU or CPU)
        images, labels = images.to(device), labels.to(device)

        # Zero out the gradients from the previous step to prevent accumulation
        optimizer.zero_grad()

        # ---------------------
        # Forward Pass
        # ---------------------
        # Pass the input data through the model
        # `outputs` contains the raw model outputs (logits) before softmax activation
        outputs = model(images).logits  

        # Compute the loss between predicted and true labels
        loss = loss_fn(outputs, labels)

        # Add current batch loss to the total loss (for calculating average loss later)
        total_loss += loss.item()

        # ---------------------
        # Backward Pass and Optimization
        # ---------------------
        # Compute gradients by backpropagation
        loss.backward()

        # Update model parameters using the optimizer
        optimizer.step()

        # ---------------------
        # Compute Accuracy
        # ---------------------
        # Get the index of the maximum logit value along dimension 1 (class prediction)
        _, predicted = torch.max(outputs, 1)  # Shape of predicted = [batch_size]

        # Count the number of correct predictions
        total_correct += (predicted == labels).sum().item()

        # Track the total number of samples processed so far
        total_samples += labels.size(0)

    # Compute the average loss over the entire training set
    avg_loss = total_loss / len(train_loader)

    # Compute the average accuracy over the entire training set
    avg_accuracy = total_correct / total_samples * 100

    # Return the average loss and accuracy
    return avg_loss, avg_accuracy

###############################################################################

# 19) test_vit_model_2

def test_vit_model_2(model, test_loader, loss_fn, device):
    """
    Function to evaluate a model on the test dataset.

    Args:
        model (torch.nn.Module): The trained neural network model.
        test_loader (torch.utils.data.DataLoader): DataLoader providing batches of test data.
        loss_fn (torch.nn.Module): Loss function to compute the error (e.g., CrossEntropyLoss).
        device (torch.device): Device to run the testing on ('cuda' or 'cpu').

    Returns:
        tuple: (avg_loss, avg_accuracy)
            - avg_loss (float): Average loss over the test dataset.
            - avg_accuracy (float): Average accuracy over the test dataset.
    """

    # ---------------------
    # Set Model to Evaluation Mode
    # ---------------------
    # In evaluation mode, dropout and batch normalization layers behave differently.
    # - Dropout layers are disabled (all neurons are active).
    # - Batch normalization uses running averages instead of batch statistics.
    model.eval()

    # Initialize variables to track total loss, correct predictions, and total samples
    total_loss = 0
    total_correct = 0
    total_samples = 0

    # ---------------------
    # Disable Gradient Calculation
    # ---------------------
    # `torch.no_grad()` prevents PyTorch from calculating and storing gradients.
    # - Reduces memory consumption and speeds up computation.
    with torch.no_grad():
        # Loop over each batch of data from the test_loader
        for images, labels in test_loader:
            # Move input data and labels to the specified device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)

            # ---------------------
            # Forward Pass
            # ---------------------
            # Pass the input data through the model
            # `outputs` contains the raw model outputs (logits) before softmax activation
            outputs = model(images).logits
            
            # Compute the loss between predicted and true labels
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            # ---------------------
            # Compute Accuracy
            # ---------------------
            # `torch.max(outputs, 1)` returns:
            #   - Values: highest logit value along dimension 1 (not used here)
            #   - Indices: index of the highest value along dimension 1 (predicted class)
            _, predicted = torch.max(outputs, 1)  

            # Count the number of correct predictions
            total_correct += (predicted == labels).sum().item()

            # Track the total number of samples processed so far
            total_samples += labels.size(0)

    # ---------------------
    # Calculate Average Loss and Accuracy
    # ---------------------
    # Average loss = total loss across all batches divided by number of batches
    avg_loss = total_loss / len(test_loader)

    # Average accuracy = total correct predictions / total samples
    avg_accuracy = total_correct / total_samples * 100

    # ---------------------
    # Return Results
    # ---------------------
    return avg_loss, avg_accuracy
    

#%%


# Prompt the user for the file path
file_path = input("Please enter the file path to the CSV file: ")
batch_size = 16

# Process the CSV file
data, labels = process_csv_fixed_id_uniform_sampling(file_path = file_path , mac_id_list = \
["00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"00:FC:BA:38:4B:02", \
"70:DB:98:9E:3A:A0", \
"70:DB:98:9E:3A:A1"], max_samples_per_mac=1000)
dataset_vit_model_1 = CustomDataset_vit_model_1(data, labels)
# Split into train and test datasets
train_size = int(0.9 * len(dataset_vit_model_1))
test_size = len(dataset_vit_model_1) - train_size
train_dataset_vit_model_1, test_dataset_vit_model_1 = random_split(dataset_vit_model_1, [train_size, test_size])
train_loader_vit_model_1 = DataLoader(train_dataset_vit_model_1, batch_size=batch_size, shuffle=True)
test_loader_vit_model_1 = DataLoader(test_dataset_vit_model_1, batch_size=batch_size, shuffle=False)



#%%
num_classes = 5
learning_rate = 5e-5
num_epochs = 50

model = vit_model_2(num_classes = num_classes)
model.to(device)

# Set up the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()


# Training the model
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Training
    train_loss, train_accuracy = train_vit_model_2(model = model, train_loader = train_loader_vit_model_1, optimizer = optimizer, loss_fn = loss_fn , device = device)
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    # Testing
    test_loss, test_accuracy = test_vit_model_2(model = model, test_loader = test_loader_vit_model_1, loss_fn = loss_fn, device = device)
    print(f"Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_accuracy:.2f}%")





#%% Training and Testing 

# Prompt the user for the file path
file_path = input("Please enter the file path to the CSV file: ")

# Process the CSV file
data, labels = process_csv_fixed_id_uniform_sampling(file_path = file_path , mac_id_list = \
["00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"00:FC:BA:38:4B:02", \
"2A:C8:A7:E1:8F:F0"], max_samples_per_mac=1000)
data = data.unsqueeze(1)
dataset = mean_norm(data)
train_loader, test_loader = train_test_loader(dataset, labels)

num_classes = 5
learning_rate = 0.001
num_epochs = 50

# Model setup
model = SimpleCNN(num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Model Training
model.train()
train(model=model, train_loader=train_loader, test_loader=test_loader, criterion=criterion, optimizer=optimizer, num_epochs=num_epochs)

# Model Testing
model.eval()
_ = test(model, test_loader)

