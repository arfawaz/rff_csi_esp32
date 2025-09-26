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


#%%

import csv  # Importing csv module to read CSV files
from collections import defaultdict  # defaultdict helps to initialize dictionary keys with default values

def count_macid_occurrences(csv_path, target_macids):
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

# Example usage (you can replace these values with your actual data)
target_macids = [
"6C:B2:AE:39:1A:A0", \
"70:0F:6A:DE:EC:A0", \
"70:0F:6A:DE:EC:A1", \
"6C:B2:AE:39:1A:A1", \
"70:0F:6A:DE:EC:A2", \
"6C:B2:AE:39:1A:A2", \
"C8:28:E5:44:3B:00", \
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"70:0F:6A:FC:51:80", \
"00:FC:BA:38:4B:02", \
"84:3D:C6:5F:5D:50"
]  # List of MAC IDs to track
csv_file = r'C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4\p4_last.csv'  # Replace with the path to your CSV file

# Run the function to count occurrences
count_macid_occurrences(csv_file, target_macids)




#%%



p4_p5_p6_all_merged_withoutcounts = 

[
"6C:B2:AE:39:1A:A0", \
"70:0F:6A:DE:EC:A0", \
"70:0F:6A:DE:EC:A1", \
"6C:B2:AE:39:1A:A1", \
"70:0F:6A:DE:EC:A2", \
"6C:B2:AE:39:1A:A2", \
"C8:28:E5:44:3B:00", \
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"70:0F:6A:FC:51:80", \
"00:FC:BA:38:4B:02", \
"84:3D:C6:5F:5D:50"
]


p4_p5_p6_all_merged_withcounts = 

6C:B2:AE:39:1A:A0: 130110
70:0F:6A:DE:EC:A0: 164017
70:0F:6A:DE:EC:A1: 150051
6C:B2:AE:39:1A:A1: 124587
70:0F:6A:DE:EC:A2: 142003
6C:B2:AE:39:1A:A2: 119773
C8:28:E5:44:3B:00: 66907
00:FC:BA:38:4B:00: 63031
00:FC:BA:38:4B:01: 57305
70:0F:6A:FC:51:80: 60413
00:FC:BA:38:4B:02: 53856
84:3D:C6:5F:5D:50: 62813
    
    
    
p4_p5_p6_all_but_last_merged_withcounts =

6C:B2:AE:39:1A:A0: 105017
70:0F:6A:DE:EC:A0: 125224
70:0F:6A:DE:EC:A1: 114182
6C:B2:AE:39:1A:A1: 100650
70:0F:6A:DE:EC:A2: 108229
6C:B2:AE:39:1A:A2: 96698
C8:28:E5:44:3B:00: 55061
00:FC:BA:38:4B:00: 54930
00:FC:BA:38:4B:01: 50323
70:0F:6A:FC:51:80: 51756
00:FC:BA:38:4B:02: 47334
84:3D:C6:5F:5D:50: 52148


p4_p5_p6_last_merged_withcounts =

6C:B2:AE:39:1A:A0: 25093
70:0F:6A:DE:EC:A0: 38793
70:0F:6A:DE:EC:A1: 35869
6C:B2:AE:39:1A:A1: 23937
70:0F:6A:DE:EC:A2: 33774
6C:B2:AE:39:1A:A2: 23075
C8:28:E5:44:3B:00: 11846
00:FC:BA:38:4B:00: 8101
00:FC:BA:38:4B:01: 6982
70:0F:6A:FC:51:80: 8657
00:FC:BA:38:4B:02: 6522
84:3D:C6:5F:5D:50: 10665
    
    

p4_all_merged_withcounts = 

6C:B2:AE:39:1A:A0: 48319
70:0F:6A:DE:EC:A0: 39753
70:0F:6A:DE:EC:A1: 37917
6C:B2:AE:39:1A:A1: 45710
70:0F:6A:DE:EC:A2: 37718
6C:B2:AE:39:1A:A2: 42993
C8:28:E5:44:3B:00: 30351
00:FC:BA:38:4B:00: 18549
00:FC:BA:38:4B:01: 16714
70:0F:6A:FC:51:80: 27188
00:FC:BA:38:4B:02: 15612
84:3D:C6:5F:5D:50: 19037



p4_all_but_last_merged_withcounts =

6C:B2:AE:39:1A:A0: 39195
70:0F:6A:DE:EC:A0: 29833
70:0F:6A:DE:EC:A1: 28371
6C:B2:AE:39:1A:A1: 36898
70:0F:6A:DE:EC:A2: 28261
6C:B2:AE:39:1A:A2: 34557
C8:28:E5:44:3B:00: 22206
00:FC:BA:38:4B:00: 16315
00:FC:BA:38:4B:01: 14554
70:0F:6A:FC:51:80: 23223
00:FC:BA:38:4B:02: 13602
84:3D:C6:5F:5D:50: 15526


p4_last_withcounts = 

6C:B2:AE:39:1A:A0: 9124
70:0F:6A:DE:EC:A0: 9920
70:0F:6A:DE:EC:A1: 9546
6C:B2:AE:39:1A:A1: 8812
70:0F:6A:DE:EC:A2: 9457
6C:B2:AE:39:1A:A2: 8436
C8:28:E5:44:3B:00: 8145
00:FC:BA:38:4B:00: 2234
00:FC:BA:38:4B:01: 2160
70:0F:6A:FC:51:80: 3965
00:FC:BA:38:4B:02: 2010
84:3D:C6:5F:5D:50: 3511
    

p5_all_merged_withcounts =

6C:B2:AE:39:1A:A0: 35637
70:0F:6A:DE:EC:A0: 67237
70:0F:6A:DE:EC:A1: 59138
6C:B2:AE:39:1A:A1: 34239
70:0F:6A:DE:EC:A2: 53732
6C:B2:AE:39:1A:A2: 33295
C8:28:E5:44:3B:00: 14896
00:FC:BA:38:4B:00: 24421
00:FC:BA:38:4B:01: 21833
70:0F:6A:FC:51:80: 13496
00:FC:BA:38:4B:02: 20190
84:3D:C6:5F:5D:50: 15276 


p5_all_but_last_merged_withcounts =

6C:B2:AE:39:1A:A0: 23891
70:0F:6A:DE:EC:A0: 55191
70:0F:6A:DE:EC:A1: 47994
6C:B2:AE:39:1A:A1: 23318
70:0F:6A:DE:EC:A2: 43233
6C:B2:AE:39:1A:A2: 22958
C8:28:E5:44:3B:00: 12144
00:FC:BA:38:4B:00: 20718
00:FC:BA:38:4B:01: 18794
70:0F:6A:FC:51:80: 9507
00:FC:BA:38:4B:02: 17400
84:3D:C6:5F:5D:50: 9291


p5_last_withcounts = 

6C:B2:AE:39:1A:A0: 11746
70:0F:6A:DE:EC:A0: 12046
70:0F:6A:DE:EC:A1: 11144
6C:B2:AE:39:1A:A1: 10921
70:0F:6A:DE:EC:A2: 10499
6C:B2:AE:39:1A:A2: 10337
C8:28:E5:44:3B:00: 2752
00:FC:BA:38:4B:00: 3703
00:FC:BA:38:4B:01: 3039
70:0F:6A:FC:51:80: 3989
00:FC:BA:38:4B:02: 2790
84:3D:C6:5F:5D:50: 5985


p6_all_merged_withcounts =

6C:B2:AE:39:1A:A0: 46154
70:0F:6A:DE:EC:A0: 57027
70:0F:6A:DE:EC:A1: 52996
6C:B2:AE:39:1A:A1: 44638
70:0F:6A:DE:EC:A2: 50553
6C:B2:AE:39:1A:A2: 43485
C8:28:E5:44:3B:00: 21660
00:FC:BA:38:4B:00: 20061
00:FC:BA:38:4B:01: 18758
70:0F:6A:FC:51:80: 19729
00:FC:BA:38:4B:02: 18054
84:3D:C6:5F:5D:50: 28500



p6_all_but_last_merged_withcounts =

6C:B2:AE:39:1A:A0: 41931
70:0F:6A:DE:EC:A0: 40200
70:0F:6A:DE:EC:A1: 37817
6C:B2:AE:39:1A:A1: 40434
70:0F:6A:DE:EC:A2: 36735
6C:B2:AE:39:1A:A2: 39183
C8:28:E5:44:3B:00: 20711
00:FC:BA:38:4B:00: 17897
00:FC:BA:38:4B:01: 16975
70:0F:6A:FC:51:80: 19026
00:FC:BA:38:4B:02: 16332
84:3D:C6:5F:5D:50: 27331


p6_last_withcounts =

6C:B2:AE:39:1A:A0: 4223
70:0F:6A:DE:EC:A0: 16827
70:0F:6A:DE:EC:A1: 15179
6C:B2:AE:39:1A:A1: 4204
70:0F:6A:DE:EC:A2: 13818
6C:B2:AE:39:1A:A2: 4302
C8:28:E5:44:3B:00: 949
00:FC:BA:38:4B:00: 2164
00:FC:BA:38:4B:01: 1783
70:0F:6A:FC:51:80: 703
00:FC:BA:38:4B:02: 1722
84:3D:C6:5F:5D:50: 1169


#%%

import pandas as pd
import os
import re
from collections import defaultdict

def extract_position(filename):
    """Extract position like 'p1', 'p2' from filename."""
    match = re.search(r"(p\d+)", filename)
    return match.group(1) if match else "unknown"

def process_mac_counts(mac_ids, file_paths):
    # Nested dictionary: {mac_id: {position: count}}
    count_dict = {mac: defaultdict(int) for mac in mac_ids}
    total_mac_counts = defaultdict(int)  # Total count across all files

    for file_path in file_paths:
        position = extract_position(os.path.basename(file_path))
        try:
            df = pd.read_csv(file_path, header=None)
            mac_column = df.iloc[:, 0]  # First column always macid

            for mac in mac_ids:
                count = (mac_column == mac).sum()
                count_dict[mac][position] += count
                total_mac_counts[mac] += count
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Create DataFrame
    positions = sorted({extract_position(os.path.basename(f)) for f in file_paths})
    result_data = []

    for mac in mac_ids:
        row = []
        for pos in positions:
            count = count_dict[mac].get(pos, 0)
            total = total_mac_counts[mac]
            percentage = (count / total * 100) if total > 0 else 0
            cell = f"{count} ({percentage:.1f}%)"
            row.append(cell)
        result_data.append(row)

    result_df = pd.DataFrame(result_data, index=mac_ids, columns=positions)
    print("\nMAC ID Occurrence Table:")
    print(result_df.to_string())
    return result_df

# Example usage

mac_ids = [
"6C:B2:AE:39:1A:A0", \
"70:0F:6A:DE:EC:A0", \
"70:0F:6A:DE:EC:A1", \
"6C:B2:AE:39:1A:A1", \
"70:0F:6A:DE:EC:A2", \
"6C:B2:AE:39:1A:A2", \
"C8:28:E5:44:3B:00", \
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"70:0F:6A:FC:51:80", \
"00:FC:BA:38:4B:02", \
"84:3D:C6:5F:5D:50"
]
file_paths = [
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4\all_but_last_merged\p4_all_but_last_merged.csv",
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p5\all_but_last_merged\p5_all_but_last_merged.csv",
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p6\all_but_last_merged\p6_all_but_last_merged.csv"
]
process_mac_counts(mac_ids, file_paths)


#%%

import os
import csv
import re
from collections import defaultdict

def extract_position_from_filename(filename):
    """
    Extracts the position (e.g., 'p1', 'p2') from the filename.
    """
    match = re.search(r'(p\d+)', filename)
    return match.group(1) if match else "unknown"

def count_valid_macid_occurrences(csv_path, target_macids):
    """
    Count valid occurrences of each MAC ID in a CSV file.

    Only rows with 128 CSI values and expected format (2 or 4 columns) are counted.
    """
    counts = defaultdict(int)
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            macid = row[0].strip()
            if len(row) == 2:
                csi_str = row[1].strip()
            elif len(row) == 4:
                csi_str = row[3].strip()
            else:
                continue
            if len(csi_str.split()) != 128:
                continue
            if macid in target_macids:
                counts[macid] += 1
    return counts

def build_macid_position_table(mac_ids, file_paths):
    """
    Constructs and prints a 2D table of MAC ID occurrences per position.
    """
    # Dictionary: macid -> {position -> count}
    count_dict = {mac: defaultdict(int) for mac in mac_ids}
    total_mac_counts = defaultdict(int)  # Total per MAC ID

    positions = set()

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        position = extract_position_from_filename(filename)
        positions.add(position)

        file_counts = count_valid_macid_occurrences(file_path, mac_ids)
        for mac in mac_ids:
            count = file_counts.get(mac, 0)
            count_dict[mac][position] += count
            total_mac_counts[mac] += count

    # Sort positions
    sorted_positions = sorted(positions)

    # Print header
    header = ["MAC ID"] + sorted_positions
    print("\t".join(header))

    for mac in mac_ids:
        row = [mac]
        for pos in sorted_positions:
            count = count_dict[mac].get(pos, 0)
            total = total_mac_counts[mac]
            percentage = (count / total * 100) if total > 0 else 0
            row.append(f"{count} ({percentage:.1f}%)")
        print("\t".join(row))

mac_ids = [
"6C:B2:AE:39:1A:A0", \
"70:0F:6A:DE:EC:A0", \
"70:0F:6A:DE:EC:A1", \
"6C:B2:AE:39:1A:A1", \
"70:0F:6A:DE:EC:A2", \
"6C:B2:AE:39:1A:A2", \
"C8:28:E5:44:3B:00", \
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"70:0F:6A:FC:51:80", \
"00:FC:BA:38:4B:02", \
"84:3D:C6:5F:5D:50"
]
file_paths = [
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4\all_but_last_merged\p4_all_but_last_merged.csv",
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p5\all_but_last_merged\p5_all_but_last_merged.csv",
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p6\all_but_last_merged\p6_all_but_last_merged.csv"
]
build_macid_position_table(mac_ids, file_paths)



#%%

import os
import csv
import re
from collections import defaultdict

def extract_position_from_filename(filename):
    match = re.search(r'(p\d+)', filename)
    return match.group(1) if match else "unknown"

def count_valid_macid_occurrences(csv_path, target_macids):
    counts = defaultdict(int)
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            macid = row[0].strip()
            if len(row) == 2:
                csi_str = row[1].strip()
            elif len(row) == 4:
                csi_str = row[3].strip()
            else:
                continue
            if len(csi_str.split()) != 128:
                continue
            if macid in target_macids:
                counts[macid] += 1
    return counts

def build_macid_position_table(mac_ids, file_paths):
    count_dict = {mac: defaultdict(int) for mac in mac_ids}
    total_mac_counts = defaultdict(int)
    positions = set()

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        position = extract_position_from_filename(filename)
        positions.add(position)
        file_counts = count_valid_macid_occurrences(file_path, mac_ids)
        for mac in mac_ids:
            count = file_counts.get(mac, 0)
            count_dict[mac][position] += count
            total_mac_counts[mac] += count

    sorted_positions = sorted(positions)
    col_width = 18  # Adjust this if you need wider spacing

    # Print header
    header = ["MAC ID".ljust(col_width)] + [pos.ljust(col_width) for pos in sorted_positions]
    print("".join(header))

    # Print rows
    for mac in mac_ids:
        row = [mac.ljust(col_width)]
        total = total_mac_counts[mac]
        for pos in sorted_positions:
            count = count_dict[mac].get(pos, 0)
            percentage = (count / total * 100) if total > 0 else 0
            cell = f"{count} ({percentage:.1f}%)"
            row.append(cell.ljust(col_width))
        print("".join(row))

mac_ids = [
"6C:B2:AE:39:1A:A0", \
"70:0F:6A:DE:EC:A0", \
"70:0F:6A:DE:EC:A1", \
"6C:B2:AE:39:1A:A1", \
"70:0F:6A:DE:EC:A2", \
"6C:B2:AE:39:1A:A2", \
"C8:28:E5:44:3B:00", \
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"70:0F:6A:FC:51:80", \
"00:FC:BA:38:4B:02", \
"84:3D:C6:5F:5D:50"
]
file_paths = [
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4\all_but_last_merged\p4_all_but_last_merged.csv",
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p5\all_but_last_merged\p5_all_but_last_merged.csv",
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p6\all_but_last_merged\p6_all_but_last_merged.csv"
]
build_macid_position_table(mac_ids, file_paths)


#%%

import os
import csv
import re
from collections import defaultdict

def extract_position_from_filename(filename):
    match = re.search(r'(p\d+)', filename)
    return match.group(1) if match else "unknown"

def count_valid_macid_occurrences(csv_path, target_macids):
    counts = defaultdict(int)
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            macid = row[0].strip()
            if len(row) == 2:
                csi_str = row[1].strip()
            elif len(row) == 4:
                csi_str = row[3].strip()
            else:
                continue
            if len(csi_str.split()) != 128:
                continue
            if macid in target_macids:
                counts[macid] += 1
    return counts

def build_macid_position_table(mac_ids, file_paths):
    count_dict = {mac: defaultdict(int) for mac in mac_ids}
    total_mac_counts = defaultdict(int)
    position_counts = defaultdict(int)  # Total counts per position (column totals)
    positions = set()

    # Count occurrences for each MAC ID and each position
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        position = extract_position_from_filename(filename)
        positions.add(position)
        file_counts = count_valid_macid_occurrences(file_path, mac_ids)
        for mac in mac_ids:
            count = file_counts.get(mac, 0)
            count_dict[mac][position] += count
            total_mac_counts[mac] += count
            position_counts[position] += count

    sorted_positions = sorted(positions)
    col_width = 18  # Adjust this if needed

    # Print header (positions)
    header = ["MAC ID".ljust(col_width)] + [pos.ljust(col_width) for pos in sorted_positions] + ["Total".ljust(col_width)]
    print("".join(header))

    # Print rows for each MAC ID
    for mac in mac_ids:
        row = [mac.ljust(col_width)]
        row_total = 0
        for pos in sorted_positions:
            count = count_dict[mac].get(pos, 0)
            percentage = (count / total_mac_counts[mac] * 100) if total_mac_counts[mac] > 0 else 0
            cell = f"{count} ({percentage:.1f}%)"
            row.append(cell.ljust(col_width))
            row_total += count
        row.append(f"{row_total}".ljust(col_width))
        print("".join(row))

    # Print column totals
    column_totals_row = ["Total".ljust(col_width)]
    total_of_totals = 0
    for pos in sorted_positions:
        total = position_counts.get(pos, 0)
        column_totals_row.append(f"{total}".ljust(col_width))
        total_of_totals += total
    column_totals_row.append(f"{total_of_totals}".ljust(col_width))
    print("".join(column_totals_row))

mac_ids = [
"6C:B2:AE:39:1A:A0", \
"70:0F:6A:DE:EC:A0", \
"70:0F:6A:DE:EC:A1", \
"6C:B2:AE:39:1A:A1", \
"70:0F:6A:DE:EC:A2", \
"6C:B2:AE:39:1A:A2", \
"C8:28:E5:44:3B:00", \
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"70:0F:6A:FC:51:80", \
"00:FC:BA:38:4B:02", \
"84:3D:C6:5F:5D:50"
]
file_paths = [
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4\all_but_last_merged\p4_all_but_last_merged.csv",
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p5\all_but_last_merged\p5_all_but_last_merged.csv",
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p6\all_but_last_merged\p6_all_but_last_merged.csv"
]
build_macid_position_table(mac_ids, file_paths)


#%%

import os
import csv
import re
from collections import defaultdict

def extract_position_from_filename(filename):
    match = re.search(r'(p\d+)', filename)
    return match.group(1) if match else "unknown"

def count_valid_macid_occurrences(csv_path, target_macids):
    counts = defaultdict(int)
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            macid = row[0].strip()
            if len(row) == 2:
                csi_str = row[1].strip()
            elif len(row) == 4:
                csi_str = row[3].strip()
            else:
                continue
            if len(csi_str.split()) != 128:
                continue
            if macid in target_macids:
                counts[macid] += 1
    return counts

def build_macid_position_table(mac_ids, file_paths):
    count_dict = {mac: defaultdict(int) for mac in mac_ids}
    total_mac_counts = defaultdict(int)
    position_counts = defaultdict(int)  # Total counts per position (column totals)
    positions = set()

    # Count occurrences for each MAC ID and each position
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        position = extract_position_from_filename(filename)
        positions.add(position)
        file_counts = count_valid_macid_occurrences(file_path, mac_ids)
        for mac in mac_ids:
            count = file_counts.get(mac, 0)
            count_dict[mac][position] += count
            total_mac_counts[mac] += count
            position_counts[position] += count

    sorted_positions = sorted(positions)
    col_width = 18  # Adjust this if needed

    # Find the minimum total across all MAC IDs
    min_total = min(total_mac_counts.values())

    # Print header (positions)
    header = ["MAC ID".ljust(col_width)] + [pos.ljust(col_width) for pos in sorted_positions] + ["Total".ljust(col_width)]
    print("".join(header))

    # Print rows for each MAC ID and adjust values based on the minimum total
    adjusted_counts = defaultdict(dict)  # Store adjusted values for each MAC ID and position
    for mac in mac_ids:
        row = [mac.ljust(col_width)]
        row_total = 0
        for pos in sorted_positions:
            count = count_dict[mac].get(pos, 0)
            percentage = (count / total_mac_counts[mac] * 100) if total_mac_counts[mac] > 0 else 0
            adjusted_value = (percentage / 100) * min_total  # Adjust the value
            adjusted_counts[mac][pos] = adjusted_value
            row_total += adjusted_value
            cell = f"{adjusted_value:.1f}"
            row.append(cell.ljust(col_width))
        row.append(f"{row_total:.1f}".ljust(col_width))
        print("".join(row))

    # Print column totals for adjusted values
    column_totals_row = ["Total".ljust(col_width)]
    total_of_totals = 0
    for pos in sorted_positions:
        total = position_counts.get(pos, 0)
        column_totals_row.append(f"{total:.1f}".ljust(col_width))
        total_of_totals += total
    column_totals_row.append(f"{total_of_totals:.1f}".ljust(col_width))
    print("".join(column_totals_row))

mac_ids = [
"6C:B2:AE:39:1A:A0", \
"70:0F:6A:DE:EC:A0", \
"70:0F:6A:DE:EC:A1", \
"6C:B2:AE:39:1A:A1", \
"70:0F:6A:DE:EC:A2", \
"6C:B2:AE:39:1A:A2", \
"C8:28:E5:44:3B:00", \
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"70:0F:6A:FC:51:80", \
"00:FC:BA:38:4B:02", \
"84:3D:C6:5F:5D:50"
]
file_paths = [
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4\all_but_last_merged\p4_all_but_last_merged.csv",
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p5\all_but_last_merged\p5_all_but_last_merged.csv",
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p6\all_but_last_merged\p6_all_but_last_merged.csv"
]
build_macid_position_table(mac_ids, file_paths)

#%%

import os
import csv
import re
from collections import defaultdict

def extract_position_from_filename(filename):
    match = re.search(r'(p\d+)', filename)
    return match.group(1) if match else "unknown"

def count_valid_macid_occurrences(csv_path, target_macids):
    counts = defaultdict(int)
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            macid = row[0].strip()
            if len(row) == 2:
                csi_str = row[1].strip()
            elif len(row) == 4:
                csi_str = row[3].strip()
            else:
                continue
            if len(csi_str.split()) != 128:
                continue
            if macid in target_macids:
                counts[macid] += 1
    return counts

def build_macid_position_table(mac_ids, file_paths):
    count_dict = {mac: defaultdict(int) for mac in mac_ids}
    total_mac_counts = defaultdict(int)
    position_counts = defaultdict(int)  # Total counts per position (column totals)
    positions = set()

    # Count occurrences for each MAC ID and each position
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        position = extract_position_from_filename(filename)
        positions.add(position)
        file_counts = count_valid_macid_occurrences(file_path, mac_ids)
        for mac in mac_ids:
            count = file_counts.get(mac, 0)
            count_dict[mac][position] += count
            total_mac_counts[mac] += count
            position_counts[position] += count

    sorted_positions = sorted(positions)
    col_width = 18  # Adjust this if needed

    # Find the minimum total across all MAC IDs
    min_total = min(total_mac_counts.values())

    # Print header (positions)
    header = ["MAC ID".ljust(col_width)] + [pos.ljust(col_width) for pos in sorted_positions] + ["Total".ljust(col_width)]
    print("Original Table (Counts & Percentages)".center(len(header) * col_width, "-"))
    print("".join(header))

    # Print original rows for each MAC ID
    for mac in mac_ids:
        row = [mac.ljust(col_width)]
        row_total = 0
        for pos in sorted_positions:
            count = count_dict[mac].get(pos, 0)
            percentage = (count / total_mac_counts[mac] * 100) if total_mac_counts[mac] > 0 else 0
            cell = f"{count} ({percentage:.1f}%)"
            row.append(cell.ljust(col_width))
            row_total += count
        row.append(f"{row_total}".ljust(col_width))
        print("".join(row))

    # Print original column totals
    column_totals_row = ["Total".ljust(col_width)]
    total_of_totals = 0
    for pos in sorted_positions:
        total = position_counts.get(pos, 0)
        column_totals_row.append(f"{total}".ljust(col_width))
        total_of_totals += total
    column_totals_row.append(f"{total_of_totals}".ljust(col_width))
    print("".join(column_totals_row))

    # Now, calculate and print the adjusted table (using minimum total)
    print("\nAdjusted Table (Based on Minimum Total)".center(len(header) * col_width, "-"))
    print("".join(header))

    # Print rows for the adjusted table
    adjusted_counts = defaultdict(dict)  # Store adjusted values for each MAC ID and position
    for mac in mac_ids:
        row = [mac.ljust(col_width)]
        row_total = 0
        for pos in sorted_positions:
            count = count_dict[mac].get(pos, 0)
            percentage = (count / total_mac_counts[mac] * 100) if total_mac_counts[mac] > 0 else 0
            adjusted_value = (percentage / 100) * min_total  # Adjust the value
            adjusted_counts[mac][pos] = adjusted_value
            row_total += adjusted_value
            cell = f"{adjusted_value:.1f}"
            row.append(cell.ljust(col_width))
        row.append(f"{row_total:.1f}".ljust(col_width))
        print("".join(row))

    # Print column totals for adjusted values
    column_totals_row = ["Total".ljust(col_width)]
    total_of_totals = 0
    for pos in sorted_positions:
        total = sum(adjusted_counts[mac][pos] for mac in mac_ids)
        column_totals_row.append(f"{total:.1f}".ljust(col_width))
        total_of_totals += total
    column_totals_row.append(f"{total_of_totals:.1f}".ljust(col_width))
    print("".join(column_totals_row))

mac_ids = [
"6C:B2:AE:39:1A:A0", \
"70:0F:6A:DE:EC:A0", \
"70:0F:6A:DE:EC:A1", \
"6C:B2:AE:39:1A:A1", \
"70:0F:6A:DE:EC:A2", \
"6C:B2:AE:39:1A:A2", \
"C8:28:E5:44:3B:00", \
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"70:0F:6A:FC:51:80", \
"00:FC:BA:38:4B:02", \
"84:3D:C6:5F:5D:50"
]
file_paths = [
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4\all_but_last_merged\p4_all_but_last_merged.csv",
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p5\all_but_last_merged\p5_all_but_last_merged.csv",
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p6\all_but_last_merged\p6_all_but_last_merged.csv"
]
build_macid_position_table(mac_ids, file_paths)

#%%

import os
import csv
import re
from collections import defaultdict

def extract_position_from_filename(filename):
    match = re.search(r'(p\d+)', filename)
    return match.group(1) if match else "unknown"

def count_valid_macid_occurrences(csv_path, target_macids):
    counts = defaultdict(int)
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            macid = row[0].strip()
            if len(row) == 2:
                csi_str = row[1].strip()
            elif len(row) == 4:
                csi_str = row[3].strip()
            else:
                continue
            if len(csi_str.split()) != 128:
                continue
            if macid in target_macids:
                counts[macid] += 1
    return counts

def build_macid_position_table(mac_ids, file_paths):
    count_dict = {mac: defaultdict(int) for mac in mac_ids}
    total_mac_counts = defaultdict(int)
    position_counts = defaultdict(int)  # Total counts per position (column totals)
    positions = set()

    # Count occurrences for each MAC ID and each position
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        position = extract_position_from_filename(filename)
        positions.add(position)
        file_counts = count_valid_macid_occurrences(file_path, mac_ids)
        for mac in mac_ids:
            count = file_counts.get(mac, 0)
            count_dict[mac][position] += count
            total_mac_counts[mac] += count
            position_counts[position] += count

    sorted_positions = sorted(positions)
    col_width = 18  # Adjust this if needed

    # Find the minimum total across all MAC IDs
    min_total = min(total_mac_counts.values())

    # Print header (positions)
    header = ["MAC ID".ljust(col_width)] + [pos.ljust(col_width) for pos in sorted_positions] + ["Total".ljust(col_width)]
    print("Original Table (Counts & Percentages)".center(len(header) * col_width, "-"))
    print("".join(header))

    # Print original rows for each MAC ID
    for mac in mac_ids:
        row = [mac.ljust(col_width)]
        row_total = 0
        for pos in sorted_positions:
            count = count_dict[mac].get(pos, 0)
            percentage = (count / total_mac_counts[mac] * 100) if total_mac_counts[mac] > 0 else 0
            cell = f"{count} ({percentage:.1f}%)"
            row.append(cell.ljust(col_width))
            row_total += count
        row.append(f"{row_total}".ljust(col_width))
        print("".join(row))

    # Print original column totals
    column_totals_row = ["Total".ljust(col_width)]
    total_of_totals = 0
    for pos in sorted_positions:
        total = position_counts.get(pos, 0)
        column_totals_row.append(f"{total}".ljust(col_width))
        total_of_totals += total
    column_totals_row.append(f"{total_of_totals}".ljust(col_width))
    print("".join(column_totals_row))

    # Now, calculate and print the adjusted table (using minimum total)
    print("\nAdjusted Table (Based on Minimum Total)".center(len(header) * col_width, "-"))
    print("".join(header))

    # Print rows for the adjusted table
    adjusted_counts = defaultdict(dict)  # Store adjusted values for each MAC ID and position
    for mac in mac_ids:
        row = [mac.ljust(col_width)]
        row_total = 0
        for pos in sorted_positions:
            count = count_dict[mac].get(pos, 0)
            percentage = (count / total_mac_counts[mac] * 100) if total_mac_counts[mac] > 0 else 0
            adjusted_value = (percentage / 100) * min_total  # Adjust the value
            adjusted_counts[mac][pos] = adjusted_value
            row_total += adjusted_value
            cell = f"{adjusted_value:.1f}"
            row.append(cell.ljust(col_width))
        row.append(f"{row_total:.1f}".ljust(col_width))
        print("".join(row))

    # Print column totals for adjusted values
    column_totals_row = ["Total".ljust(col_width)]
    total_of_totals = 0
    for pos in sorted_positions:
        total = sum(adjusted_counts[mac][pos] for mac in mac_ids)
        column_totals_row.append(f"{total:.1f}".ljust(col_width))
        total_of_totals += total
    column_totals_row.append(f"{total_of_totals:.1f}".ljust(col_width))
    print("".join(column_totals_row))

    # Return both original and adjusted counts
    return count_dict, adjusted_counts


mac_ids = [
"6C:B2:AE:39:1A:A0", \
"70:0F:6A:DE:EC:A0", \
"70:0F:6A:DE:EC:A1", \
"6C:B2:AE:39:1A:A1", \
"70:0F:6A:DE:EC:A2", \
"6C:B2:AE:39:1A:A2", \
"C8:28:E5:44:3B:00", \
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"70:0F:6A:FC:51:80", \
"00:FC:BA:38:4B:02", \
"84:3D:C6:5F:5D:50"
]
    

# file_paths = [
#     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\p4_last.csv", \
#     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\p5_last.csv", \
#     r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\p6_last.csv"
# ]


file_paths = [
    #r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4\all_but_last_merged\p4_all_but_last_merged.csv", \
    #r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p5\all_but_last_merged\p5_all_but_last_merged.csv", \
    #r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p6\all_but_last_merged\p6_all_but_last_merged.csv"
    #r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p5\all_but_last_merged\p5_all_but_last_merged.csv"
    #r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p6\all_but_last_merged\p6_all_but_last_merged.csv"
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p7\all_but_last_merged\p7_all_but_last_merged.csv"
]    
a_,b__ = build_macid_position_table(mac_ids, file_paths)


#%%

import csv
import random
import os
import re
from collections import defaultdict

def extract_position_from_filename(filepath):
    """
    Extracts the position identifier (e.g., 'p4', 'p5') from the filename part of a file path.

    Args:
        filepath (str): The full path to the CSV file.

    Returns:
        str: The extracted position string like 'p4'. Returns 'unknown' if not found.
    """
    # Get just the filename (e.g., 'p4_all_but_last_merged.csv')
    filename = os.path.basename(filepath)

    # Search for the position pattern (e.g., p4, p12)
    match = re.search(r'(p\d+)', filename)

    # Return matched position or 'unknown'
    return match.group(1) if match else "unknown"


def sample_macid_rows(csv_path, target_macids, adjusted_counts, position, output_folder):
    """
    Sample rows from a CSV file based on the number of occurrences needed for each MAC ID.
    
    Parameters:
    - csv_path (str): Path to the CSV file to be read and sampled.
    - target_macids (list of str): A list of MAC IDs for which we want to sample rows.
    - adjusted_counts (dict): A dictionary containing the adjusted number of samples per MAC ID for each position.
    - position (str): The position (e.g., 'p1', 'p2') from the filename which is used to get the required number of samples for each MAC ID.
    - output_folder (str): The folder path where the new sampled CSV file will be saved.
    
    Returns:
    - None: The function writes the sampled rows to a new CSV file in the specified output folder.
    """
    # Create a dictionary to hold the rows for each MAC ID
    macid_rows = defaultdict(list)
    
    # Open the input CSV file and read its content
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        
        # Loop through each row in the CSV file
        for row in reader:
            if not row:  # Skip any empty rows
                continue
            macid = row[0].strip()  # Extract the MAC ID (first column)
            if macid in target_macids:
                macid_rows[macid].append(row)  # Group rows by MAC ID
    
    # List to store the sampled rows
    sampled_rows = []
    
    # Sample rows for each MAC ID based on adjusted counts
    for macid in target_macids:
        if macid in macid_rows:
            # Calculate how many rows to sample for this MAC ID at the current position
            num_samples = int(adjusted_counts[macid][position])  # Round down to nearest integer
            print(f'num_samples is: {num_samples}')
            if len(macid_rows[macid]) >= num_samples:
                # If there are enough rows, sample randomly
                sampled_rows.extend(random.sample(macid_rows[macid], num_samples))
            else:
                # If there aren't enough rows, include all available rows for this MAC ID
                sampled_rows.extend(macid_rows[macid])
    
    # Ensure the output folder exists, create it if necessary
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate the output file path based on the position
    new_file_name = os.path.join(output_folder, f"sampled_{position}.csv")
    
    # Write the sampled rows to the new CSV file
    with open(new_file_name, 'w', newline='') as new_file:
        writer = csv.writer(new_file)
        writer.writerows(sampled_rows)  # Write all sampled rows to the new file
    
    print(f"Sampled data for position {position} saved in {new_file_name}")


def create_sampled_csv_files(mac_ids, file_paths, adjusted_counts, output_folder):
    """
    Generate sampled CSV files based on the adjusted counts for each MAC ID across multiple CSV files.
    
    Parameters:
    - mac_ids (list of str): List of MAC IDs for which we want to sample rows.
    - file_paths (list of str): List of paths to the CSV files containing data for different positions (e.g., 'p1', 'p2').
    - adjusted_counts (dict): A dictionary containing the number of samples needed for each MAC ID per position (e.g., 'p1', 'p2').
    - output_folder (str): The folder path where the new sampled CSV files will be saved.
    
    Returns:
    - None: The function generates and saves new CSV files in the specified output folder.
    """
    # Loop over each CSV file path (position) and sample the data
    for file_path in file_paths:
        # Extract the position from the file name (e.g., 'p1', 'p2')
        position = extract_position_from_filename(file_path)
        print(f"position is: {position}")
        
        # Call the function to sample data for the current position
        sample_macid_rows(file_path, mac_ids, adjusted_counts, position, output_folder)



# List of MAC IDs we are interested in sampling
mac_ids = [
"6C:B2:AE:39:1A:A0", \
"70:0F:6A:DE:EC:A0", \
"70:0F:6A:DE:EC:A1", \
"6C:B2:AE:39:1A:A1", \
"70:0F:6A:DE:EC:A2", \
"6C:B2:AE:39:1A:A2", \
"C8:28:E5:44:3B:00", \
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"70:0F:6A:FC:51:80", \
"00:FC:BA:38:4B:02", \
"84:3D:C6:5F:5D:50"
]

# List of file paths for the different CSV files (representing different positions like p1, p2, etc.)
file_paths = [
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4\all_but_last_merged\p4_all_but_last_merged.csv",
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p5\all_but_last_merged\p5_all_but_last_merged.csv",
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p6\all_but_last_merged\p6_all_but_last_merged.csv"
]

# Dictionary containing adjusted counts for each MAC ID at different positions (e.g., p1, p2)
_,adjusted_counts = build_macid_position_table(mac_ids, file_paths)

# Specify the folder where the sampled CSV files will be saved
output_folder = r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\test"

# Call the function to create sampled CSV files
create_sampled_csv_files(mac_ids, file_paths, adjusted_counts, output_folder)


#%% updated mac filter logic
import csv
import random
import os
import re
from collections import defaultdict

# Extracts position identifier like 'p4' or 'p5' from a file path
def extract_position_from_filename(filepath):
    filename = os.path.basename(filepath)  # Get file name from full path
    match = re.search(r'(p\d+)', filename)  # Search for 'p<number>'
    return match.group(1) if match else "unknown"  # Return match or 'unknown'

# Sample rows from a CSV file based on adjusted count, only if valid CSI row
def sample_macid_rows(csv_path, target_macids, adjusted_counts, position, output_folder):
    """
    Sample rows from CSV for each MAC ID at a given position,
    filtering by CSI string length (must be 128).
    """
    macid_rows = defaultdict(list)  # Group rows by MAC ID

    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue  # Skip empty rows

            macid = row[0].strip()

            # Determine CSI string column depending on row length
            if len(row) == 2:
                csi_str = row[1].strip()
            elif len(row) == 4:
                csi_str = row[3].strip()
            else:
                continue  # Skip malformed rows

            if len(csi_str.split()) != 128:
                continue  # Skip rows with invalid CSI length

            if macid in target_macids:
                macid_rows[macid].append(row)  # Keep valid rows only

    sampled_rows = []  # Final list of sampled rows

    for macid in target_macids:
        if macid in macid_rows:
            num_samples = int(adjusted_counts[macid][position])  # How many to sample
            print(f'num_samples is: {num_samples}')
            if len(macid_rows[macid]) >= num_samples:
                sampled_rows.extend(random.sample(macid_rows[macid], num_samples))
            else:
                sampled_rows.extend(macid_rows[macid])  # Not enough rows, take all

    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    new_file_name = os.path.join(output_folder, f"sampled_{position}.csv")

    with open(new_file_name, 'w', newline='') as new_file:
        writer = csv.writer(new_file)
        writer.writerows(sampled_rows)  # Write all sampled rows

    print(f"Sampled data for position {position} saved in {new_file_name}")

# Wrapper to handle multiple CSV files
def create_sampled_csv_files(mac_ids, file_paths, adjusted_counts, output_folder):
    for file_path in file_paths:
        position = extract_position_from_filename(file_path)  # Get position (e.g., p4)
        print(f"position is: {position}")
        sample_macid_rows(file_path, mac_ids, adjusted_counts, position, output_folder)


# ========================
# ==== USER INPUT AREA ===
# ========================

# List of MAC IDs of interest
mac_ids = [
    "6C:B2:AE:39:1A:A0",
    "70:0F:6A:DE:EC:A0",
    "70:0F:6A:DE:EC:A1",
    "6C:B2:AE:39:1A:A1",
    "70:0F:6A:DE:EC:A2",
    "6C:B2:AE:39:1A:A2",
    "C8:28:E5:44:3B:00",
    "00:FC:BA:38:4B:00",
    "00:FC:BA:38:4B:01",
    "70:0F:6A:FC:51:80",
    "00:FC:BA:38:4B:02",
    "84:3D:C6:5F:5D:50"
]

# File paths for each position
file_paths = [
    #r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\p4_last.csv", \
    #r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\p5_last.csv", \
    #r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\p6_last.csv"
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4\all_but_last_merged\p4_all_but_last_merged.csv"
]

# Path where output sampled files will go
output_folder = r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\test"

# Function assumed to exist (not defined here). You must define it elsewhere.
# It should return a tuple: (original_counts, adjusted_counts)
_, adjusted_counts = build_macid_position_table(mac_ids, file_paths)

# Start the sampling process
create_sampled_csv_files(mac_ids, file_paths, adjusted_counts, output_folder)



#%%

sampled_file_paths =  [ \
    #r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\test\test_set\sampled_p4.csv", \
    #r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\test\test_set\sampled_p5.csv", \
    #r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\test\test_set\sampled_p6.csv" \
    #r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\test\sampled_p5.csv"
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\test\sampled_p4.csv"
]
_,adjusted_counts_sampled = build_macid_position_table(mac_ids, sampled_file_paths)


#%% two paths new code

import os
import csv
import re
from collections import defaultdict

def extract_position_from_filename(filename):
    match = re.search(r'(p\d+)', filename)
    return match.group(1) if match else "unknown"

def count_valid_macid_occurrences(csv_path, target_macids):
    counts = defaultdict(int)
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            macid = row[0].strip()
            if len(row) == 2:
                csi_str = row[1].strip()
            elif len(row) == 4:
                csi_str = row[3].strip()
            else:
                continue
            if len(csi_str.split()) != 128:
                continue
            if macid in target_macids:
                counts[macid] += 1
    return counts

def build_and_adjust_macid_tables(mac_ids, file_paths_percent_base, file_paths_adjust_target,min_samples = None):
    col_width = 18

    def build_counts(file_paths, label):
        count_dict = {mac: defaultdict(int) for mac in mac_ids}
        total_mac_counts = defaultdict(int)
        position_counts = defaultdict(int)
        positions = set()

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            position = extract_position_from_filename(filename)
            positions.add(position)
            file_counts = count_valid_macid_occurrences(file_path, mac_ids)
            for mac in mac_ids:
                count = file_counts.get(mac, 0)
                count_dict[mac][position] += count
                total_mac_counts[mac] += count
                position_counts[position] += count

        sorted_positions = sorted(positions)

        print(f"\n{label} Table (Original Counts and Percentages)".center(len(sorted_positions) * col_width + 3 * col_width, "-"))
        header = ["MAC ID".ljust(col_width)] + [pos.ljust(col_width) for pos in sorted_positions] + ["Total".ljust(col_width)]
        print("".join(header))

        for mac in mac_ids:
            row = [mac.ljust(col_width)]
            row_total = 0
            for pos in sorted_positions:
                count = count_dict[mac].get(pos, 0)
                percentage = (count / total_mac_counts[mac] * 100) if total_mac_counts[mac] > 0 else 0
                row.append(f"{count} ({percentage:.1f}%)".ljust(col_width))
                row_total += count
            row.append(str(row_total).ljust(col_width))
            print("".join(row))

        total_row = ["Total".ljust(col_width)]
        grand_total = 0
        for pos in sorted_positions:
            total = position_counts[pos]
            total_row.append(str(total).ljust(col_width))
            grand_total += total
        total_row.append(str(grand_total).ljust(col_width))
        print("".join(total_row))

        return count_dict, total_mac_counts, sorted_positions

    # Build counts for the base set (for percentage calculation)
    base_counts, base_total_mac_counts, base_positions = build_counts(file_paths_percent_base, "BASE SET")

    # Calculate percentage per MAC ID across base set
    base_percentages = defaultdict(dict)
    for mac in mac_ids:
        total = base_total_mac_counts[mac]
        for pos in base_positions:
            count = base_counts[mac][pos]
            base_percentages[mac][pos] = (count / total * 100) if total > 0 else 0

    # Build counts for the target set (for applying adjusted sampling)
    target_counts, target_total_mac_counts, target_positions = build_counts(file_paths_adjust_target, "TARGET SET")

    # Get the minimum total across all MACs in the target set
    if min_samples == None:
        min_total_target = min(target_total_mac_counts.values())
    else:
        min_total_target = min_samples

    # Adjust the target set based on base percentages and min total of target
    print(f"\nAdjusted Counts for TARGET SET Based on BASE SET Percentages (Scaled to min total = {min_total_target})".center(len(target_positions) * col_width + 3 * col_width, "-"))
    header = ["MAC ID".ljust(col_width)] + [pos.ljust(col_width) for pos in target_positions] + ["Total".ljust(col_width)]
    print("".join(header))

    adjusted_target_counts = defaultdict(dict)
    for mac in mac_ids:
        row = [mac.ljust(col_width)]
        total_adjusted = 0
        for pos in target_positions:
            percentage = base_percentages[mac].get(pos, 0)
            adjusted_value = (percentage / 100) * min_total_target
            adjusted_target_counts[mac][pos] = adjusted_value
            row.append(f"{adjusted_value:.1f}".ljust(col_width))
            total_adjusted += adjusted_value
        row.append(f"{total_adjusted:.1f}".ljust(col_width))
        print("".join(row))

    total_row = ["Total".ljust(col_width)]
    grand_total = 0
    for pos in target_positions:
        total = sum(adjusted_target_counts[mac][pos] for mac in mac_ids)
        total_row.append(f"{total:.1f}".ljust(col_width))
        grand_total += total
    total_row.append(f"{grand_total:.1f}".ljust(col_width))
    print("".join(total_row))

    return adjusted_target_counts

#%%

# mac_ids = [ ... ]  # your list of MAC IDs

# base_files = [
#     r"path\to\p4_all_but_last_merged.csv",
#     r"path\to\p5_all_but_last_merged.csv",
#     r"path\to\p6_all_but_last_merged.csv"
# ]

# target_files = [
#     r"path\to\p4_last.csv",
#     r"path\to\p5_last.csv",
#     r"path\to\p6_last.csv"
# ]

mac_ids = [
    "6C:B2:AE:39:1A:A0",
    "70:0F:6A:DE:EC:A0",
    "70:0F:6A:DE:EC:A1",
    "6C:B2:AE:39:1A:A1",
    "70:0F:6A:DE:EC:A2",
    "6C:B2:AE:39:1A:A2",
    "C8:28:E5:44:3B:00",
    "00:FC:BA:38:4B:00",
    "00:FC:BA:38:4B:01",
    "70:0F:6A:FC:51:80",
    "00:FC:BA:38:4B:02",
    "84:3D:C6:5F:5D:50"
]


base_files = [
    #r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4\all_but_last_merged\p4_all_but_last_merged.csv", \
    #r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p5\all_but_last_merged\p5_all_but_last_merged.csv" \
    #r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p6\all_but_last_merged\p6_all_but_last_merged.csv"
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\test\sampled_p4.csv"
]  

target_files = [
    #r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\p4_last.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\p4_last.csv" \
    #r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\p6_last.csv"
]


  



adjusted_counts_test = build_and_adjust_macid_tables(mac_ids, base_files, target_files,min_samples=1000)


#%% updated mac filter logic repeated again for test set making

import csv
import random
import os
import re
from collections import defaultdict

# Extracts position identifier like 'p4' or 'p5' from a file path
def extract_position_from_filename(filepath):
    filename = os.path.basename(filepath)  # Get file name from full path
    match = re.search(r'(p\d+)', filename)  # Search for 'p<number>'
    return match.group(1) if match else "unknown"  # Return match or 'unknown'

# Sample rows from a CSV file based on adjusted count, only if valid CSI row
def sample_macid_rows(csv_path, target_macids, adjusted_counts, position, output_folder):
    """
    Sample rows from CSV for each MAC ID at a given position,
    filtering by CSI string length (must be 128).
    """
    macid_rows = defaultdict(list)  # Group rows by MAC ID

    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue  # Skip empty rows

            macid = row[0].strip()

            # Determine CSI string column depending on row length
            if len(row) == 2:
                csi_str = row[1].strip()
            elif len(row) == 4:
                csi_str = row[3].strip()
            else:
                continue  # Skip malformed rows

            if len(csi_str.split()) != 128:
                continue  # Skip rows with invalid CSI length

            if macid in target_macids:
                macid_rows[macid].append(row)  # Keep valid rows only

    sampled_rows = []  # Final list of sampled rows

    for macid in target_macids:
        if macid in macid_rows:
            num_samples = int(adjusted_counts[macid][position])  # How many to sample
            print(f'num_samples is: {num_samples}')
            if len(macid_rows[macid]) >= num_samples:
                sampled_rows.extend(random.sample(macid_rows[macid], num_samples))
            else:
                sampled_rows.extend(macid_rows[macid])  # Not enough rows, take all

    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    new_file_name = os.path.join(output_folder, f"sampled_{position}.csv")

    with open(new_file_name, 'w', newline='') as new_file:
        writer = csv.writer(new_file)
        writer.writerows(sampled_rows)  # Write all sampled rows

    print(f"Sampled data for position {position} saved in {new_file_name}")

# Wrapper to handle multiple CSV files
def create_sampled_csv_files(mac_ids, file_paths, adjusted_counts, output_folder):
    for file_path in file_paths:
        position = extract_position_from_filename(file_path)  # Get position (e.g., p4)
        print(f"position is: {position}")
        sample_macid_rows(file_path, mac_ids, adjusted_counts_test, position, output_folder)


# ========================
# ==== USER INPUT AREA ===
# ========================

# List of MAC IDs of interest
mac_ids = [
    "6C:B2:AE:39:1A:A0",
    "70:0F:6A:DE:EC:A0",
    "70:0F:6A:DE:EC:A1",
    "6C:B2:AE:39:1A:A1",
    "70:0F:6A:DE:EC:A2",
    "6C:B2:AE:39:1A:A2",
    "C8:28:E5:44:3B:00",
    "00:FC:BA:38:4B:00",
    "00:FC:BA:38:4B:01",
    "70:0F:6A:FC:51:80",
    "00:FC:BA:38:4B:02",
    "84:3D:C6:5F:5D:50"
]

# File paths for each position
file_paths = [
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\p4_last.csv", \
    #r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\p5_last.csv" \
    #r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\p6_last.csv"
]

# Path where output sampled files will go
output_folder = r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\Individual_positions\p4_p5_p6\test\test_set"

# Function assumed to exist (not defined here). You must define it elsewhere.
# It should return a tuple: (original_counts, adjusted_counts)
#_, adjusted_counts = build_macid_position_table(mac_ids, file_paths)

# Start the sampling process
create_sampled_csv_files(mac_ids, file_paths, adjusted_counts_test, output_folder)

#%%

import os
import re

def extract_position_from_filename(filepath):
    """
    Extracts the position identifier (e.g., 'p4', 'p5') from the filename part of a file path.

    Args:
        filepath (str): The full path to the CSV file.

    Returns:
        str: The extracted position string like 'p4'. Returns 'unknown' if not found.
    """
    # Get just the filename (e.g., 'p4_all_but_last_merged.csv')
    filename = os.path.basename(filepath)

    # Search for the position pattern (e.g., p4, p12)
    match = re.search(r'(p\d+)', filename)

    # Return matched position or 'unknown'
    return match.group(1) if match else "unknown"

for file_path in file_paths:
    # Extract the position from the file name (e.g., 'p1', 'p2')
    position = extract_position_from_filename(file_path)
    print(f"position is: {position}")


#%%

import os
import re

def rename_files_with_sample_offset(folder_path, offset):
    pattern = re.compile(r'(class\d+_sample)(\d+)(\.csv)', re.IGNORECASE)

    for filename in os.listdir(folder_path):
        print(f"Checking: {filename}")
        match = pattern.match(filename)
        if match:
            prefix, sample_num_str, suffix = match.groups()
            new_sample_num = int(sample_num_str) + offset
            new_filename = f"{prefix}{new_sample_num}{suffix}"
            
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)

            print(f"Renaming: {filename} → {new_filename}")
            os.rename(old_path, new_path)
        else:
            print("  → No match")

# Example usage
folder_path = r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\caa_authentication\test_bed_data\mumcu preprocessing\processes_data\training_2m_t2"
offset = 3572
rename_files_with_sample_offset(folder_path, offset)

#%%

import csv
import torch
import random
import os
import re

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
            real = float(csi_values[i])
            imag = float(csi_values[i + 1])
            csi_tensor.append([real, imag])
        except ValueError:
            return None  # Skip rows with invalid numeric values
    return torch.tensor(csi_tensor)

def extract_distance_from_filename(filename):
    """
    Extracts distance in meters from a filename like 'turf_04m_060_degree.csv' -> 4.0
    """
    match = re.search(r'_([0-9]{2})m_', filename)
    if match:
        return float(match.group(1))
    return None

def process_csv_files_by_distance(file_paths, mac_id, max_samples_per_file=50000):
    """
    Processes multiple CSV files to extract CSI data for a specific MAC address,
    and labels each sample with the distance (in meters) extracted from the filename.
    """
    data = []
    labels = []

    for file_path in file_paths:
        distance = extract_distance_from_filename(os.path.basename(file_path))
        if distance is None:
            continue  # Skip file if distance not found

        valid_entries = []

        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) != 4:
                    continue
                current_mac_id, _, _, csi_row = row
                if current_mac_id != mac_id:
                    continue
                csi_tensor = parse_csi_data(csi_row)
                if csi_tensor is not None:
                    valid_entries.append(csi_tensor)

        sample_size = min(len(valid_entries), max_samples_per_file)
        if sample_size == 0:
            continue

        sampled_entries = random.sample(valid_entries, sample_size)
        data.extend(sampled_entries)
        labels.extend([distance] * sample_size)

    if data:
        csi_data = torch.stack(data)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return csi_data, labels_tensor
    else:
        return None, None

#%%

file_paths = [
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_01m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_01m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_01m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_01m_180_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_02m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_02m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_02m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_02m_180_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_03m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_03m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_03m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_03m_180_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_04m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_04m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_04m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_04m_180_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_05m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_05m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_05m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_05m_180_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_06m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_06m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_06m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_06m_180_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_07m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_07m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_07m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_07m_180_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_08m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_08m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_08m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_08m_180_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_09m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_09m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_09m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_09m_180_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_10m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_10m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_10m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_10m_180_degree.csv"
]
mac_id = "34:5F:45:A9:A4:19"

csi_tensor, labels_tensor = process_csv_files_by_distance(file_paths = file_paths, mac_id = mac_id, max_samples_per_file=2000)

print(csi_tensor.shape)         # [N, 64, 2]
print(labels_tensor.shape)  # [N]
print(labels_tensor.unique())  # e.g., tensor([ 4., 10.])
#%%

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim

# ==== Step 1: Prepare the dataset (csi_tensor: [N, 64, 2], labels_tensor: [N]) ====

# Flatten CSI: (N, 64, 2) → (N, 128)
X = csi_tensor.view(csi_tensor.size(0), -1)  # Shape: (N, 128)
y = labels_tensor.view(-1, 1)                # Shape: (N, 1)

# Remove all-zero CSI samples
nonzero_mask = (X.abs().sum(dim=1) != 0)
X = X[nonzero_mask]
y = y[nonzero_mask]

# Compute mean and std
X_mean = X.mean(dim=0)
X_std = X.std(dim=0)

# Replace std=0 with 1 to avoid divide-by-zero
X_std[X_std == 0] = 1.0

# Normalize
X = (X - X_mean) / X_std

# Check for NaNs or Infs (optional debug)
assert not torch.isnan(X).any(), "NaN detected in input features"
assert not torch.isinf(X).any(), "Inf detected in input features"

# Train-test split: 80% / 20%
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# ==== Step 2: Define the MLP Model ====

class CSIRegressor(nn.Module):
    def __init__(self, input_dim=128):
        super(CSIRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Predict scalar distance
        )

    def forward(self, x):
        return self.net(x)

# ==== Step 3: Training ====

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CSIRegressor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)

    train_mse = running_loss / train_size
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train MSE: {train_mse:.4f}")

# ==== Step 4: Evaluation ====

model.eval()
with torch.no_grad():
    total_loss = 0.0
    all_preds = []aaqqqqqqqqq
    all_targets = []

    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        total_loss += loss.item() * batch_x.size(0)
        all_preds.append(preds.cpu())
        all_targets.append(batch_y.cpu())

    test_mse = total_loss / test_size
    print(f"Test MSE: {test_mse:.4f}")

# Optional: convert predictions to NumPy for plotting or analysis
predictions = torch.cat(all_preds).numpy().flatten()
targets = torch.cat(all_targets).numpy().flatten()

#%%

# Print 10 sample predictions vs ground truth
print("\nSample predictions vs actual distances (in meters):")
for i in range(min(100, len(predictions))):
    print(f"Predicted: {predictions[i]:.2f} m, Actual: {targets[i]:.2f} m")


#%% SECTION 1: CSI CSV Processing

import csv
import torch
import random
import os
import re

def parse_csi_data(csi_row):
    """
    Parses a single row of CSI data (expected 128 space-separated values)
    into a 64x2 PyTorch tensor of real and imaginary parts.
    """
    csi_values = csi_row.split()
    if len(csi_values) != 128:
        return None  # Skip if not exactly 64 complex values
    csi_tensor = []
    for i in range(0, 128, 2):
        try:
            real = float(csi_values[i])
            imag = float(csi_values[i + 1])
            csi_tensor.append([real, imag])
        except ValueError:
            return None  # Skip row if conversion fails
    return torch.tensor(csi_tensor)

def extract_distance_from_filename(filename):
    """
    Extracts the distance in meters from filenames like 'turf_04m_060_degree.csv'.
    """
    match = re.search(r'_([0-9]{2})m_', filename)
    if match:
        return float(match.group(1))
    return None

def process_csv_files_by_distance(file_paths, mac_id, max_samples_per_file=50000):
    """
    Processes multiple CSV files to extract CSI data for a specific MAC address.
    Each sample is labeled with the distance extracted from the filename.

    Parameters:
        file_paths: list of CSV file paths
        mac_id: the MAC ID of interest
        max_samples_per_file: how many samples to retain from each file

    Returns:
        csi_data: Tensor of shape [N, 64, 2]
        labels_tensor: Tensor of distances [N]
    """
    data = []
    labels = []

    for file_path in file_paths:
        distance = extract_distance_from_filename(os.path.basename(file_path))
        if distance is None:
            continue  # Skip file if distance cannot be parsed

        valid_entries = []

        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) != 4:
                    continue  # Expect 4 columns (MAC, RSSI, NF, CSI)
                current_mac_id, _, _, csi_row = row
                if current_mac_id != mac_id:
                    continue  # Skip if MAC doesn't match
                csi_tensor = parse_csi_data(csi_row)
                if csi_tensor is not None:
                    valid_entries.append(csi_tensor)

        # Uniform random sampling of valid entries
        sample_size = min(len(valid_entries), max_samples_per_file)
        if sample_size == 0:
            continue
        sampled_entries = random.sample(valid_entries, sample_size)

        data.extend(sampled_entries)
        labels.extend([distance] * sample_size)

    if data:
        csi_data = torch.stack(data)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return csi_data, labels_tensor
    else:
        return None, None

#%% SECTION 2: Load CSI dataset from files

# List of all CSV file paths
file_paths = [
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_01m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_01m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_01m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_01m_180_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_02m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_02m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_02m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_02m_180_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_03m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_03m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_03m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_03m_180_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_04m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_04m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_04m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_04m_180_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_05m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_05m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_05m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_05m_180_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_06m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_06m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_06m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_06m_180_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_07m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_07m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_07m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_07m_180_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_08m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_08m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_08m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_08m_180_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_09m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_09m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_09m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_09m_180_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_10m_000_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_10m_060_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_10m_120_degree.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\USF\SEMESTER 1 - FALL 23\DIRECTED RESEARCH\projects_on_git\rff_csi_esp32\csi_data_collected\trilaterlation\turf_data\turf_10m_180_degree.csv"
]  # (your full list of file paths)

# Target MAC ID to filter rows by
mac_id = "34:5F:45:A9:A4:19"

# Load CSI data and distance labels
csi_tensor, labels_tensor = process_csv_files_by_distance(
    file_paths=file_paths, mac_id=mac_id, max_samples_per_file=2000
)

# Print shapes and unique distance labels
print(csi_tensor.shape)         # Expected: (N, 64, 2)
print(labels_tensor.shape)      # Expected: (N,)
print(labels_tensor.unique())   # List of unique distances

#%% SECTION 3: Preprocessing for training/testing

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim

# Flatten CSI data: from (N, 64, 2) → (N, 128)
X = csi_tensor.view(csi_tensor.size(0), -1)  # Shape: (N, 128)
y = labels_tensor.view(-1, 1)                # Shape: (N, 1)

# Remove samples with all-zero CSI (no signal)
nonzero_mask = (X.abs().sum(dim=1) != 0)
X = X[nonzero_mask]
y = y[nonzero_mask]

# Normalize inputs (standardization)
X_mean = X.mean(dim=0)
X_std = X.std(dim=0)
X_std[X_std == 0] = 1.0  # Prevent divide-by-zero
X = (X - X_mean) / X_std

# Sanity checks
assert not torch.isnan(X).any(), "NaN detected in input features"
assert not torch.isinf(X).any(), "Inf detected in input features"

# Split dataset: 80% training, 20% testing
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

#%% SECTION 4: Define MLP Model

class CSIRegressor(nn.Module):
    def __init__(self, input_dim=128):
        super(CSIRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output: predicted distance (scalar)
        )

    def forward(self, x):
        return self.net(x)

#%% SECTION 5: Training Loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CSIRegressor().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)

    train_mse = running_loss / train_size
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train MSE: {train_mse:.4f}")

#%% SECTION 6: Evaluation

model.eval()
with torch.no_grad():
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        total_loss += loss.item() * batch_x.size(0)
        all_preds.append(preds.cpu())
        all_targets.append(batch_y.cpu())

    test_mse = total_loss / test_size
    print(f"Test MSE: {test_mse:.4f}")

# Convert predictions and targets to NumPy for analysis
predictions = torch.cat(all_preds).numpy().flatten()
targets = torch.cat(all_targets).numpy().flatten()

#%% SECTION 7: Display Predictions vs Actual

# Print first 100 predicted vs actual distances
print("\nSample predictions vs actual distances (in meters):")
for i in range(min(100, len(predictions))):
    print(f"Predicted: {predictions[i]:.2f} m, Actual: {targets[i]:.2f} m")

#%%

import numpy as np
import matplotlib.pyplot as plt

def trilaterate_2d(anchors, distances):
    """
    Estimate the 2D position of an unknown point using trilateration from multiple anchors.
    Uses least-squares approximation if more than 3 anchors are given.

    Parameters:
        anchors   : list of (x, y) coordinates of reference anchors
        distances : list of distances to each anchor

    Returns:
        pos       : estimated (x, y) position of the unknown point
    """
    x1, y1 = anchors[0]   # Use first anchor as reference
    d1 = distances[0]

    A = []
    b = []

    # Build linear system using relative distances
    for i in range(1, len(anchors)):
        xi, yi = anchors[i]
        di = distances[i]
        A.append([2 * (xi - x1), 2 * (yi - y1)])
        b.append(d1**2 - di**2 - x1**2 + xi**2 - y1**2 + yi**2)

    A = np.array(A)
    b = np.array(b)

    # Solve Ax = b using least squares
    pos, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return pos

def compute_error_grid(anchors, distance_error=1.0, grid_size=100, xlim=(0, 10), ylim=(0, 10)):
    """
    Generate a heatmap of position estimation error across a 2D region.

    Parameters:
        anchors        : list of (x, y) reference points
        distance_error : fixed error added to all distance measurements (worst-case)
        grid_size      : resolution of grid (NxN)
        xlim, ylim     : bounds of the area to evaluate

    Returns:
        x_vals, y_vals : 1D arrays of X and Y grid positions
        error_grid     : 2D array of position errors (shape: grid_size x grid_size)
    """
    x_vals = np.linspace(xlim[0], xlim[1], grid_size)
    y_vals = np.linspace(ylim[0], ylim[1], grid_size)
    error_grid = np.zeros((grid_size, grid_size))

    # Iterate over each true position in the 2D grid
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            true_pos = np.array([x, y])

            # Compute true distances to all anchors
            true_distances = [np.linalg.norm(true_pos - np.array(anchor)) for anchor in anchors]

            # Add fixed error to simulate worst-case deviation
            noisy_distances = [d + distance_error for d in true_distances]

            # Estimate position using trilateration
            estimated_pos = trilaterate_2d(anchors, noisy_distances)

            # Compute Euclidean error between estimated and true position
            error = np.linalg.norm(estimated_pos - true_pos)

            # Store error at corresponding grid cell
            error_grid[j, i] = error  # note: row is Y, column is X

    return x_vals, y_vals, error_grid

def plot_error_heatmap(x_vals, y_vals, error_grid, anchors, distance_error):
    """
    Plot a heatmap of position errors over a 2D region.

    Parameters:
        x_vals, y_vals : X and Y axis grid values
        error_grid     : 2D array of position errors
        anchors        : list of anchor coordinates
        distance_error : fixed distance error used (for title)
    """
    plt.figure(figsize=(10, 8))

    # Contour heatmap
    plt.contourf(x_vals, y_vals, error_grid, levels=50, cmap='inferno')
    plt.colorbar(label='Position Error (meters)')

    # Plot anchor locations
    anchors_np = np.array(anchors)
    plt.scatter(anchors_np[:, 0], anchors_np[:, 1], 
                c='cyan', s=100, label='Anchors', edgecolors='black', marker='X')

    # Labels and layout
    plt.title(f"Position Error Heatmap (+{distance_error}m Distance Error)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# =============================
# Example usage
# =============================

# Define 3 anchors in an equilateral triangle formation
anchors = [(0, 0), (10, 0), (5, 8.66)]

# Set simulation parameters
distance_error = 1.0         # worst-case additive error to all distances
grid_size = 100              # grid resolution (e.g., 100x100)
xlim = (-5, 15)              # X-axis plotting range
ylim = (-5, 15)              # Y-axis plotting range

# Compute position errors over grid
x_vals, y_vals, error_grid = compute_error_grid(anchors, distance_error, grid_size, xlim, ylim)

# Plot the heatmap
plot_error_heatmap(x_vals, y_vals, error_grid, anchors, distance_error)

#%%

import numpy as np
import matplotlib.pyplot as plt

def trilaterate_2d(anchors, distances):
    """
    Perform 2D trilateration using least squares.
    
    Parameters:
        anchors   : List of (x, y) tuples for reference anchor points
        distances : List of noisy distances from unknown point to each anchor
        
    Returns:
        Estimated position (x, y) of the unknown node
    """
    # Use the first anchor as a reference
    x1, y1 = anchors[0]
    d1 = distances[0]

    A = []
    b = []

    for i in range(1, len(anchors)):
        xi, yi = anchors[i]
        di = distances[i]

        # Linearized form of trilateration equations
        A.append([2 * (xi - x1), 2 * (yi - y1)])
        b.append(d1**2 - di**2 - x1**2 + xi**2 - y1**2 + yi**2)

    A = np.array(A)
    b = np.array(b)

    # Solve the linear system A*x = b using least squares
    pos, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return pos

def compute_error_grid_random_noise(anchors, max_error=1.0, grid_size=100, xlim=(0, 10), ylim=(0, 10)):
    """
    Compute a grid of positioning errors using trilateration with random distance noise.
    
    Parameters:
        anchors    : List of anchor positions (tuples)
        max_error  : Max error added/subtracted from each distance measurement
        grid_size  : Grid resolution (grid_size x grid_size)
        xlim, ylim : Tuple indicating the range of true positions on X and Y axes
        
    Returns:
        x_vals     : X-axis values (1D array)
        y_vals     : Y-axis values (1D array)
        error_grid : 2D array of errors at each (x, y) position
    """
    x_vals = np.linspace(xlim[0], xlim[1], grid_size)
    y_vals = np.linspace(ylim[0], ylim[1], grid_size)
    error_grid = np.zeros((grid_size, grid_size))

    # For each true position on the grid
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            true_pos = np.array([x, y])

            # Step 1: Compute true distances to each anchor
            true_distances = [np.linalg.norm(true_pos - np.array(anchor)) for anchor in anchors]

            # Step 2: Add random noise in range [-max_error, +max_error]
            noisy_distances = [d + np.random.uniform(-max_error, max_error) for d in true_distances]

            # Step 3: Perform trilateration to estimate position
            estimated_pos = trilaterate_2d(anchors, noisy_distances)

            # Step 4: Compute Euclidean error between estimated and true position
            error = np.linalg.norm(estimated_pos - true_pos)

            # Store error in grid (note: j is Y index, i is X index)
            error_grid[j, i] = error

    return x_vals, y_vals, error_grid

def plot_error_heatmap(x_vals, y_vals, error_grid, anchors, max_error):
    """
    Visualizes the position error as a heatmap.
    
    Parameters:
        x_vals     : 1D array of X positions
        y_vals     : 1D array of Y positions
        error_grid : 2D array of position errors
        anchors    : Anchor coordinates for display
        max_error  : Used in plot title to indicate noise level
    """
    plt.figure(figsize=(10, 8))
    
    # Heatmap plot using contour fill
    plt.contourf(x_vals, y_vals, error_grid, levels=50, cmap='inferno')
    plt.colorbar(label='Position Error (meters)')

    # Plot anchors
    anchors_np = np.array(anchors)
    plt.scatter(
        anchors_np[:, 0], anchors_np[:, 1],
        c='cyan', s=100, label='Anchors',
        edgecolors='black', marker='X'
    )

    # Labels and aesthetics
    plt.title(f"Position Error Heatmap (Random Distance Error ±{max_error}m)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# =============================
# Example usage
# =============================

# Define 3 anchor nodes in a triangle
anchors = [(0, 0), (10, 0), (5, 8.66)]  # Equilateral triangle layout

# Simulation parameters
max_error = 1.0          # Distance errors range from -1m to +1m
grid_size = 100          # Grid resolution
xlim = (-5, 15)          # X range for evaluation
ylim = (-5, 15)          # Y range for evaluation

# Run error simulation
x_vals, y_vals, error_grid = compute_error_grid_random_noise(
    anchors=anchors,
    max_error=max_error,
    grid_size=grid_size,
    xlim=xlim,
    ylim=ylim
)

# Plot the error heatmap
plot_error_heatmap(x_vals, y_vals, error_grid, anchors, max_error)


#%%

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
    
    # Format output for easy copy-pasting as a Python list with raw string literals
    if csv_file_paths:
        print("[")
        for i, path in enumerate(csv_file_paths):
            if i < len(csv_file_paths) - 1:
                print(f'    r"{path}", \\')
            else:
                print(f'    r"{path}"')
        print("]")

# Example usage:
# get_n_csv_filepaths("your/folder/path", 5)

#%%

#%% get_n_csv_filepaths()

folder_path = r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1"
n = 1000  # Number of files you want to retrieve

get_n_csv_filepaths(folder_path, n)

#%%

[
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\119_02_may_25_p1_01_10_04_10.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\p1_last.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\30_02_mar_25_p1_05_15_.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\32_03_mar_25_p1_05_00_06_00.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\35_04_mar_25_p1_04_00_05_00.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\58_12_mar_25_p1_08_55_09_55.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\95_08_apr_25_p1_03_30_06_40.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\96_08_apr_25_p1_06_40_8_50.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\97_09_apr_25_p1_04_50_08_30.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\98_10_apr_25_p1_01_45_07_10.csv"
]


[
    "C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\119_02_may_25_p1_01_10_04_10.csv", \
    "C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\p1_last.csv", \
    "C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\30_02_mar_25_p1_05_15_.csv", \
    "C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\32_03_mar_25_p1_05_00_06_00.csv", \
    "C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\35_04_mar_25_p1_04_00_05_00.csv", \
    "C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\58_12_mar_25_p1_08_55_09_55.csv", \
    "C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\95_08_apr_25_p1_03_30_06_40.csv", \
    "C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\96_08_apr_25_p1_06_40_8_50.csv", \
    "C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\97_09_apr_25_p1_04_50_08_30.csv", \
    "C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\98_10_apr_25_p1_01_45_07_10.csv"
]
    
#%%


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSI Autoencoder + Complex-Valued CNN (PyTorch)
- Input CSI shape: (batch, 2, 64)  # 2 = [real, imag], 64 subcarriers
- Pipeline:
    1) Pretrain autoencoder (MSE) to denoise CSI
    2) Train complex CNN classifier on AE-denoised CSI (or raw, toggle below)
    3) Evaluate test accuracy

Assumptions:
- process_csv_fixed_id_uniform_sampling_rssi(...) is available in your environment.
- labels are already correct (0..C-1) and of dtype torch.long.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


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

def process_csv_fixed_id_uniform_sampling_rssi(file_path, mac_id_list, max_samples_per_mac=50000):
    """
    Processes a CSV file to extract CSI data for specific MAC addresses and assigns labels based on their order in mac_id_list.
    Instead of selecting the first max_samples_per_mac entries, this function selects uniformly from all available entries.
    """
    mac_entries = {mac: [] for mac in mac_id_list}  # Store all CSI data for each MAC
    
    # Read CSV file and collect all valid CSI entries for each MAC
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) != 2 and len(row) != 4:
                continue  # Skip invalid rows (only support 2-column or 4-column rows)
            current_mac_id, csi_row = None, None

            # Check which column contains the CSI data (second or fourth)
            if len(row) == 2:
                current_mac_id, csi_row = row
            elif len(row) == 4:
                current_mac_id, _, _, csi_row = row  # Extract CSI row from the fourth column

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
        data_ = data_.squeeze()
        labels_ = torch.tensor(labels, dtype=torch.long)
        return data_, labels_
    else:
        return None, None  # Return None if no valid data was processed


# ========= Data utilities =========

def train_test_loader(data, labels, train_percent=0.8, batch_size=64, shuffle=True):
    dataset = TensorDataset(data, labels)
    train_size = int(train_percent * len(dataset))
    test_size  = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,  drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader

def ensure_shape_2x64(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure CSI tensor is shaped (N, 2, 64).
    Accepts (N, 64, 2) or (N, 2, 64). Returns (N, 2, 64).
    """
    if x.dim() != 3:
        raise ValueError(f"Expected data of dim 3 (N,*,*), got shape {tuple(x.shape)}")
    N, A, B = x.shape
    if A == 2 and B == 64:
        return x
    if A == 64 and B == 2:
        return x.permute(0, 2, 1).contiguous()
    raise ValueError(f"Unsupported input shape {tuple(x.shape)}; need (N,64,2) or (N,2,64).")

def zscore_per_sample(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-sample, per-channel z-score across subcarriers (length=64).
    Input/Output: (N, 2, 64)
    """
    mean = x.mean(dim=-1, keepdim=True)
    std  = x.std(dim=-1, keepdim=True) + eps
    return (x - mean) / std

# ========= Models =========

class CSIAutoencoder(nn.Module):
    """
    1D Conv Autoencoder
    Input:  (B, 2, 64)
    Latent: vector of size latent_dim
    Output: (B, 2, 64) reconstructed
    """
    def __init__(self, latent_dim=32):
        super().__init__()
        self.enc_conv1 = nn.Conv1d(2, 16, kernel_size=3, padding=1)   # -> (B,16,64)
        self.enc_conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)  # -> (B,32,64)
        self.enc_flat  = nn.Flatten()                                 # -> (B, 32*64)
        self.enc_fc    = nn.Linear(32*64, latent_dim)                 # -> (B, latent_dim)

        self.dec_fc    = nn.Linear(latent_dim, 32*64)
        self.dec_unflat= nn.Unflatten(1, (32, 64))                    # -> (B,32,64)
        self.dec_conv1 = nn.Conv1d(32, 16, kernel_size=3, padding=1)  # -> (B,16,64)
        self.dec_conv2 = nn.Conv1d(16, 2,  kernel_size=3, padding=1)  # -> (B,2,64)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        z = self.enc_fc(self.enc_flat(x))
        return z

    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        x = self.dec_unflat(x)
        x = F.relu(self.dec_conv1(x))
        x = self.dec_conv2(x)  # regression output
        return x

    def forward(self, x):
        z = self.encode(x)
        xr = self.decode(z)
        return xr, z

# ---- Complex ops (implemented with real conv pairs) ----

def split_complex_channels(x: torch.Tensor):
    """
    Input x: (B, 2*C, L) with interleaved channels [Re0, Im0, Re1, Im1, ...]
    Returns (xr, xi): each (B, C, L)
    """
    xr = x[:, 0::2, :]
    xi = x[:, 1::2, :]
    return xr, xi

def interleave_real_imag(xr: torch.Tensor, xi: torch.Tensor):
    """
    xr, xi: (B, C, L) -> (B, 2*C, L) interleaved
    """
    B, C, L = xr.shape
    out = torch.stack([xr, xi], dim=2)  # (B, C, 2, L)
    out = out.view(B, 2*C, L)
    return out

class ComplexConv1D(nn.Module):
    """
    Complex 1D convolution via real-valued pairs.
    Input: (B, 2*C_in, L); Output: (B, 2*C_out, L)
    y_real = x_real ⊛ W_real - x_imag ⊛ W_imag
    y_imag = x_real ⊛ W_imag + x_imag ⊛ W_real
    """
    def __init__(self, cin_complex: int, cout_complex: int, kernel_size=3, padding=1):
        super().__init__()
        self.wr = nn.Conv1d(cin_complex, cout_complex, kernel_size, padding=padding, bias=True)
        self.wi = nn.Conv1d(cin_complex, cout_complex, kernel_size, padding=padding, bias=True)

    def forward(self, x):
        xr, xi = split_complex_channels(x)      # (B, C_in, L) each
        yr = self.wr(xr) - self.wi(xi)
        yi = self.wr(xi) + self.wi(xr)
        y  = interleave_real_imag(yr, yi)       # (B, 2*C_out, L)
        return y

class ComplexBlock(nn.Module):
    """
    ComplexConv1D -> (per-part ReLU) -> optional dropout
    """
    def __init__(self, cin_complex, cout_complex, kernel_size=3, padding=1, p_drop=0.0):
        super().__init__()
        self.cconv = ComplexConv1D(cin_complex, cout_complex, kernel_size, padding)
        self.dropout = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        y = self.cconv(x)
        yr, yi = split_complex_channels(y)
        yr = F.relu(yr)
        yi = F.relu(yi)
        y  = interleave_real_imag(yr, yi)
        y  = self.dropout(y)
        return y

class ComplexCNNClassifier(nn.Module):
    """
    Complex CNN classifier operating on (B, 2, 64) complex CSI.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.block1 = ComplexBlock(cin_complex=1,  cout_complex=8,  kernel_size=3, padding=1, p_drop=0.1)  # -> (B,16,64)
        self.block2 = ComplexBlock(cin_complex=8,  cout_complex=16, kernel_size=3, padding=1, p_drop=0.1)  # -> (B,32,64)
        self.block3 = ComplexBlock(cin_complex=16, cout_complex=16, kernel_size=3, padding=1, p_drop=0.1)  # -> (B,32,64)
        self.gap = nn.AdaptiveAvgPool1d(1)  # (B, 32, 1)
        self.fc  = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (B, 2, 64)  (2 = real, imag)
        x = self.block1(x)  # (B, 16, 64)
        x = self.block2(x)  # (B, 32, 64)
        x = self.block3(x)  # (B, 32, 64)
        x = self.gap(x).squeeze(-1)  # (B, 32)
        logits = self.fc(x)          # (B, C)
        return logits

# ========= Training =========

def pretrain_autoencoder(model, train_loader, test_loader, device, epochs=25, lr=1e-3, weight_decay=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = float("inf")
    best_state = None

    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, _ in train_loader:
            xb = xb.to(device)
            xr, _ = model(xb)
            loss = F.mse_loss(xr, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)

        model.eval()
        te_loss = 0.0
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                xr, _ = model(xb)
                loss = F.mse_loss(xr, xb)
                te_loss += loss.item() * xb.size(0)

        tr_loss /= len(train_loader.dataset)
        te_loss /= len(test_loader.dataset)
        print(f"[AE] Epoch {ep:03d} | train MSE: {tr_loss:.6f} | test MSE: {te_loss:.6f}")

        if te_loss < best:
            best = te_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def evaluate_classifier(forward_fn, loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = forward_fn(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return 100.0 * correct / max(total, 1)

def train_classifier(classifier, ae, train_loader, test_loader, device,
                     epochs=40, lr=1e-3, weight_decay=1e-4,
                     denoise_with_ae=True, finetune_ae=False):
    """
    Train classifier. If denoise_with_ae=True, the input to the classifier is AE(x).
    If finetune_ae=True, AE is updated jointly; else AE is frozen during classifier training.
    """
    params = list(classifier.parameters())
    if denoise_with_ae and finetune_ae:
        params += list(ae.parameters())

    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    best_acc = -1.0
    best_state_cls = None
    best_state_ae  = None

    for ep in range(1, epochs+1):
        classifier.train()
        ae.train() if (denoise_with_ae and finetune_ae) else ae.eval()

        tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if denoise_with_ae:
                if finetune_ae:
                    xr, _ = ae(xb)       # gradients flow to AE
                    x_in = xr
                else:
                    with torch.no_grad():
                        xr, _ = ae(xb)   # AE frozen
                    x_in = xr
            else:
                x_in = xb

            logits = classifier(x_in)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += loss.item() * xb.size(0)

        tr_loss /= len(train_loader.dataset)

        # closures for evaluation
        if denoise_with_ae:
            if finetune_ae:
                forward_eval = lambda x: classifier(ae(x)[0])
            else:
                forward_eval = lambda x: classifier(ae(x)[0])
        else:
            forward_eval = lambda x: classifier(x)

        train_acc = evaluate_classifier(forward_eval, train_loader, device)
        test_acc  = evaluate_classifier(forward_eval, test_loader,  device)
        print(f"[CLS] Epoch {ep:03d} | loss: {tr_loss:.4f} | train acc: {train_acc:.2f}% | test acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_state_cls = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
            if denoise_with_ae and finetune_ae:
                best_state_ae = {k: v.cpu().clone() for k, v in ae.state_dict().items()}

    if best_state_cls is not None:
        classifier.load_state_dict(best_state_cls)
    if best_state_ae is not None:
        ae.load_state_dict(best_state_ae)

    return classifier, ae, best_acc



import os  # if not already imported

def loader_from_csv(file_path: str,
                    mac_id_list,
                    max_samples_per_mac=100000,
                    batch_size=64,
                    normalize=True):
    """
    Loads a CSV using your process_csv_fixed_id_uniform_sampling_rssi(),
    makes a DataLoader for evaluation.
    Returns: DataLoader, num_classes
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"CSV not found: {file_path}")

    data2, labels2 = process_csv_fixed_id_uniform_sampling_rssi(
        file_path=file_path,
        mac_id_list=mac_id_list,            # IMPORTANT: same order as training
        max_samples_per_mac=max_samples_per_mac,
    )

    if not isinstance(data2, torch.Tensor):
        data2 = torch.tensor(data2)
    if not isinstance(labels2, torch.Tensor):
        labels2 = torch.tensor(labels2)

    data2 = data2.float()
    labels2 = labels2.long()
    data2 = ensure_shape_2x64(data2)
    if normalize:
        data2 = zscore_per_sample(data2)

    num_classes2 = int(labels2.max().item() + 1) if labels2.numel() > 0 else 0
    ds2 = TensorDataset(data2, labels2)
    loader2 = DataLoader(ds2, batch_size=batch_size, shuffle=False, drop_last=False)
    return loader2, num_classes2


# ========= Main =========

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ----- Load your data -----
    file_path = input("Please enter the file path to the CSV file: ")
    # Import your loader if not already imported in your environment:
    # from your_module import process_csv_fixed_id_uniform_sampling_rssi
    data, labels = process_csv_fixed_id_uniform_sampling_rssi(
        file_path=file_path,
        mac_id_list=[
            "00:FC:BA:38:4B:00", \
            "00:FC:BA:38:4B:01", \
            "00:FC:BA:38:4B:02", \
            "6C:B2:AE:39:1A:A0", \
            "6C:B2:AE:39:1A:A1", \
            "70:0F:6A:DE:EC:A0", \
            "70:0F:6A:DE:EC:A1", \
            "70:0F:6A:DE:EC:A2" \
            
        ],
        max_samples_per_mac=10000
    )
    print("Done data loading")

    # ----- Tensors & shapes -----
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    data = data.float()
    labels = labels.long()   # you said labels are already correct

    data = ensure_shape_2x64(data)   # -> (N,2,64)
    data = zscore_per_sample(data)   # normalization per sample/channel

    num_classes = int(labels.max().item() + 1)
    print(f"Detected num_classes = {num_classes}")

    train_loader, test_loader = train_test_loader(data, labels, train_percent=0.8, batch_size=64)

    # ----- Models -----
    ae  = CSIAutoencoder(latent_dim=32).to(device)
    clf = ComplexCNNClassifier(num_classes=num_classes).to(device)

    # ----- Pretrain AE -----
    print("\n=== Pretraining Autoencoder (reconstruction) ===")
    ae = pretrain_autoencoder(ae, train_loader, test_loader, device,
                              epochs=25, lr=1e-3, weight_decay=0.0)

    # ----- Train classifier -----
    print("\n=== Training Complex CNN Classifier (with AE denoise) ===")
    clf, ae, best_acc = train_classifier(
        classifier=clf,
        ae=ae,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=40,
        lr=1e-3,
        weight_decay=1e-4,
        denoise_with_ae=False,   # set False to train on raw CSI
        finetune_ae=False       # set True to jointly fine-tune AE
    )

    # ----- Final test -----
    with torch.no_grad():
        forward_eval = (lambda x: clf(ae(x)[0]))  # using AE output
        final_test_acc = evaluate_classifier(forward_eval, test_loader, device)
    print(f"\n=== Final Test Accuracy: {final_test_acc:.2f}% ===")
    
        # ----- Optional external test CSV -----
    ext_path = input(
        "\n(Optional) Enter external test CSV path (leave blank to skip): "
    ).strip()
    
    if ext_path:
        try:
            # Use the SAME mac_id_list you used for training to keep labels consistent
            mac_ids = [
                "00:FC:BA:38:4B:00", \
                "00:FC:BA:38:4B:01", \
                "00:FC:BA:38:4B:02", \
                "6C:B2:AE:39:1A:A0", \
                "6C:B2:AE:39:1A:A1", \
                "70:0F:6A:DE:EC:A0", \
                "70:0F:6A:DE:EC:A1", \
                "70:0F:6A:DE:EC:A2" \
            ]
            ext_loader, ext_classes = loader_from_csv(
                ext_path,
                mac_id_list=mac_ids,
                max_samples_per_mac=100000,
                batch_size=64,
                normalize=True,
            )
    
            if ext_classes != num_classes:
                print(
                    f"[WARN] external test set has {ext_classes} classes but training had {num_classes}."
                    " Make sure your CSV loader maps labels consistently (same mac_id_list order)."
                )
    
            # with torch.no_grad():
            #     forward_eval_ext = (lambda x: clf(ae(x)[0]))  # AE-denoised
            #     ext_acc = evaluate_classifier(forward_eval_ext, ext_loader, device)
            # print(f"=== External Test Accuracy ({os.path.basename(ext_path)}): {ext_acc:.2f}% ===")
            with torch.no_grad():
                acc_raw = evaluate_classifier(lambda x: clf(x),        ext_loader, device)
                acc_ae  = evaluate_classifier(lambda x: clf(ae(x)[0]), ext_loader, device)
            
            print(f"External Test Accuracy (RAW):        {acc_raw:.2f}%")
            print(f"External Test Accuracy (AE-denoised): {acc_ae:.2f}%")

        except Exception as e:
            print(f"[ERROR] Could not evaluate external test CSV: {e}")


if __name__ == "__main__":
    try:
        main()
    except NameError as e:
        print("\nIt looks like process_csv_fixed_id_uniform_sampling_rssi(...) isn't imported in this file.")
        print("Please import it (or replace the call) and run again.\n")
        raise

#C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\Systematic collection final until Jul 31 2025-selected\individual_positions\p1\test



#%% with latent space
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSI Autoencoder + Latent-MLP Classifier (PyTorch)
- Input CSI shape: (batch, 2, 64)  # 2 = [real, imag], 64 subcarriers
- Pipeline:
    1) Pretrain autoencoder (MSE) to denoise / compress CSI
    2) Train MLP classifier on AE latent vector z
    3) Evaluate test accuracy (on z)

Assumptions:
- process_csv_fixed_id_uniform_sampling_rssi(...) is available (provided below).
- labels are already correct (0..C-1) and dtype torch.long.
"""

import os
import csv
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

# ------------------------- CSV utilities (yours) -------------------------

def parse_csi_data(csi_row):
    """
    Parses a single row of CSI data into a 64x2 PyTorch tensor [magnitude, angle].
    Expects 128 whitespace-separated numbers per row.
    """
    csi_values = csi_row.split()
    if len(csi_values) != 128:
        return None
    csi_tensor = []
    for i in range(0, 128, 2):
        try:
            magnitude = float(csi_values[i])
            angle = float(csi_values[i + 1])
            csi_tensor.append([magnitude, angle])
        except ValueError:
            return None
    return torch.tensor(csi_tensor)

def process_csv_fixed_id_uniform_sampling_rssi(file_path, mac_id_list, max_samples_per_mac=50000):
    """
    Collects CSI rows for MACs in mac_id_list, uniformly samples up to max_samples_per_mac,
    and labels by mac_id_list index.
    Returns:
      data:   (N, 64, 2) tensor
      labels: (N,)       LongTensor in 0..C-1
    """
    mac_entries = {mac: [] for mac in mac_id_list}
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) not in (2, 4):
                continue
            if len(row) == 2:
                current_mac_id, csi_row = row
            else:
                current_mac_id, _, _, csi_row = row
            if current_mac_id not in mac_id_list:
                continue
            csi_tensor = parse_csi_data(csi_row)
            if csi_tensor is not None:
                mac_entries[current_mac_id].append(csi_tensor)

    data, labels = [], []
    mac_id_to_label = {mac: i for i, mac in enumerate(mac_id_list)}
    for mac, entries in mac_entries.items():
        sample_size = min(len(entries), max_samples_per_mac)
        if sample_size == 0:
            continue
        sampled = random.sample(entries, sample_size)
        data.extend(sampled)
        labels.extend([mac_id_to_label[mac]] * sample_size)

    if len(data) == 0:
        return None, None

    data_ = torch.stack(data).squeeze()       # (N, 64, 2)
    labels_ = torch.tensor(labels, dtype=torch.long)
    return data_, labels_

# ------------------------- Data helpers -------------------------

def train_test_loader(data, labels, train_percent=0.8, batch_size=64, shuffle=True):
    dataset = TensorDataset(data, labels)
    train_size = int(train_percent * len(dataset))
    test_size  = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,  drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader

def ensure_shape_2x64(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure CSI tensor is (N, 2, 64). Accepts (N, 64, 2) or (N, 2, 64).
    """
    if x.dim() != 3:
        raise ValueError(f"Expected data of dim 3 (N,*,*), got {tuple(x.shape)}")
    N, A, B = x.shape
    if A == 2 and B == 64:
        return x
    if A == 64 and B == 2:
        return x.permute(0, 2, 1).contiguous()
    raise ValueError(f"Unsupported input shape {tuple(x.shape)}; need (N,64,2) or (N,2,64).")

def zscore_per_sample(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-sample, per-channel z-score across subcarriers.
    Input/Output: (N, 2, 64)
    """
    mean = x.mean(dim=-1, keepdim=True)
    std  = x.std(dim=-1, keepdim=True) + eps
    return (x - mean) / std

# ------------------------- Models -------------------------

class CSIAutoencoder(nn.Module):
    """
    Conv1D Autoencoder: (B, 2, 64) -> z (B, latent_dim) -> recon (B, 2, 64)
    """
    def __init__(self, latent_dim=32):
        super().__init__()
        self.enc_conv1 = nn.Conv1d(2, 16, kernel_size=3, padding=1)   # (B,16,64)
        self.enc_conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)  # (B,32,64)
        self.enc_flat  = nn.Flatten()                                 # (B, 32*64)
        self.enc_fc    = nn.Linear(32*64, latent_dim)                 # (B, latent_dim)

        self.dec_fc    = nn.Linear(latent_dim, 32*64)
        self.dec_unflat= nn.Unflatten(1, (32, 64))                    # (B,32,64)
        self.dec_conv1 = nn.Conv1d(32, 16, kernel_size=3, padding=1)  # (B,16,64)
        self.dec_conv2 = nn.Conv1d(16, 2,  kernel_size=3, padding=1)  # (B,2,64)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        z = self.enc_fc(self.enc_flat(x))
        return z

    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        x = self.dec_unflat(x)
        x = F.relu(self.dec_conv1(x))
        x = self.dec_conv2(x)  # regression target
        return x

    def forward(self, x):
        z = self.encode(x)
        xr = self.decode(z)
        return xr, z

class LatentMLPClassifier(nn.Module):
    """
    Classifier that takes AE latent vector z: (B, latent_dim) -> logits (B, C)
    """
    def __init__(self, latent_dim: int, num_classes: int, hidden=(128, 64), p_drop=0.1):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(p_drop)]
            in_dim = h
        layers += [nn.Linear(in_dim, num_classes)]  # no activation; CrossEntropyLoss expects logits
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

# ------------------------- Training / Eval -------------------------

def pretrain_autoencoder(model, train_loader, test_loader, device, epochs=25, lr=1e-3, weight_decay=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = float("inf")
    best_state = None

    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, _ in train_loader:
            xb = xb.to(device)
            xr, _ = model(xb)
            loss = F.mse_loss(xr, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)

        model.eval()
        te_loss = 0.0
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                xr, _ = model(xb)
                loss = F.mse_loss(xr, xb)
                te_loss += loss.item() * xb.size(0)

        tr_loss /= len(train_loader.dataset)
        te_loss /= len(test_loader.dataset)
        print(f"[AE] Epoch {ep:03d} | train MSE: {tr_loss:.6f} | test MSE: {te_loss:.6f}")

        if te_loss < best:
            best = te_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

@torch.no_grad()
def evaluate_classifier(forward_fn, loader, device):
    correct, total = 0, 0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        logits = forward_fn(xb)              # forward_fn should accept (B,2,64)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total   += yb.numel()
    return 100.0 * correct / max(total, 1)

def train_classifier_on_latent(clf, ae, train_loader, test_loader, device,
                               epochs=40, lr=1e-3, weight_decay=1e-4,
                               finetune_encoder=False):
    """
    Train MLP classifier on z = ae.encode(x).
    If finetune_encoder=True, encoder weights are updated jointly.
    """
    # Only the encoder is needed from AE during this stage
    ae.train() if finetune_encoder else ae.eval()

    params = list(clf.parameters())
    if finetune_encoder:
        # add ONLY encoder params to optimizer (not the decoder)
        enc_params = [p for n, p in ae.named_parameters() if n.startswith("enc_")]
        params += enc_params

    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    best_acc = -1.0
    best_state = None
    best_ae_state = None

    for ep in range(1, epochs+1):
        clf.train()
        if finetune_encoder:
            ae.train()

        tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)

            if finetune_encoder:
                z = ae.encode(xb)                 # gradients flow into encoder
            else:
                with torch.no_grad():
                    z = ae.encode(xb)             # freeze encoder

            logits = clf(z)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += loss.item() * xb.size(0)

        tr_loss /= len(train_loader.dataset)

        # Eval closures
        forward_eval = lambda x: clf(ae.encode(x))
        train_acc = evaluate_classifier(forward_eval, train_loader, device)
        test_acc  = evaluate_classifier(forward_eval, test_loader,  device)
        print(f"[CLS(z)] Epoch {ep:03d} | loss: {tr_loss:.4f} | train acc: {train_acc:.2f}% | test acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}
            if finetune_encoder:
                best_ae_state = {k: v.cpu().clone() for k, v in ae.state_dict().items()}

    if best_state is not None:
        clf.load_state_dict(best_state)
    if best_ae_state is not None:
        ae.load_state_dict(best_ae_state)

    return clf, ae, best_acc

# ------------------------- Helper for external CSV to DataLoader -------------------------

def loader_from_csv(file_path: str,
                    mac_id_list,
                    max_samples_per_mac=100000,
                    batch_size=64,
                    normalize=True):
    """
    Loads a CSV via process_csv_fixed_id_uniform_sampling_rssi(), returns (DataLoader, num_classes)
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"CSV not found: {file_path}")

    data2, labels2 = process_csv_fixed_id_uniform_sampling_rssi(
        file_path=file_path,
        mac_id_list=mac_id_list,
        max_samples_per_mac=max_samples_per_mac,
    )
    if not isinstance(data2, torch.Tensor):   data2 = torch.tensor(data2)
    if not isinstance(labels2, torch.Tensor): labels2 = torch.tensor(labels2)

    data2 = data2.float()
    labels2 = labels2.long()
    data2 = ensure_shape_2x64(data2)
    if normalize:
        data2 = zscore_per_sample(data2)

    num_classes2 = int(labels2.max().item() + 1) if labels2.numel() > 0 else 0
    ds2 = TensorDataset(data2, labels2)
    loader2 = DataLoader(ds2, batch_size=batch_size, shuffle=False, drop_last=False)
    return loader2, num_classes2

# ------------------------- Main -------------------------

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ----- Load your training CSV -----
    file_path = input("Please enter the file path to the CSV file: ")
    data, labels = process_csv_fixed_id_uniform_sampling_rssi(
        file_path=file_path,
        mac_id_list=[
            "00:FC:BA:38:4B:00",
            "00:FC:BA:38:4B:01",
            "00:FC:BA:38:4B:02",
            "6C:B2:AE:39:1A:A0",
            "6C:B2:AE:39:1A:A1",
            "70:0F:6A:DE:EC:A0",
            "70:0F:6A:DE:EC:A1",
            "70:0F:6A:DE:EC:A2"
        ],
        max_samples_per_mac=10000
    )
    print("Done data loading")

    # ----- Tensors & shapes -----
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    data = data.float()
    labels = labels.long()   # labels already correct

    data = ensure_shape_2x64(data)   # (N,2,64)
    data = zscore_per_sample(data)   # normalization

    num_classes = int(labels.max().item() + 1)
    print(f"Detected num_classes = {num_classes}")

    train_loader, test_loader = train_test_loader(data, labels, train_percent=0.8, batch_size=64)

    # ----- Models -----
    latent_dim = 32
    ae  = CSIAutoencoder(latent_dim=latent_dim).to(device)
    clf = LatentMLPClassifier(latent_dim=latent_dim, num_classes=num_classes).to(device)

    # ----- Pretrain AE (reconstruction) -----
    print("\n=== Pretraining Autoencoder (reconstruction) ===")
    ae = pretrain_autoencoder(ae, train_loader, test_loader, device,
                              epochs=25, lr=1e-3, weight_decay=0.0)

    # ----- Train classifier on latent z -----
    print("\n=== Training MLP Classifier on AE latent z ===")
    clf, ae, best_acc = train_classifier_on_latent(
        clf, ae,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=40,
        lr=1e-3,
        weight_decay=1e-4,
        finetune_encoder=False   # set True to fine-tune encoder jointly
    )

    # ----- Final test on the held-out split -----
    with torch.no_grad():
        forward_eval = lambda x: clf(ae.encode(x))  # z-path
        final_test_acc = evaluate_classifier(forward_eval, test_loader, device)
    print(f"\n=== Final Test Accuracy (latent z): {final_test_acc:.2f}% ===")

    # ----- Optional external test CSV -----
    ext_path = input("\n(Optional) Enter external test CSV path (leave blank to skip): ").strip()
    if ext_path:
        try:
            mac_ids = [
                "00:FC:BA:38:4B:00",
                "00:FC:BA:38:4B:01",
                "00:FC:BA:38:4B:02",
                "6C:B2:AE:39:1A:A0",
                "6C:B2:AE:39:1A:A1",
                "70:0F:6A:DE:EC:A0",
                "70:0F:6A:DE:EC:A1",
                "70:0F:6A:DE:EC:A2"
            ]
            ext_loader, ext_classes = loader_from_csv(
                ext_path,
                mac_id_list=mac_ids,
                max_samples_per_mac=100000,
                batch_size=64,
                normalize=True,
            )
            if ext_classes != num_classes:
                print(f"[WARN] external set has {ext_classes} classes; training had {num_classes}. "
                      "Ensure same mac_id_list order/coverage.")

            with torch.no_grad():
                ext_acc = evaluate_classifier(lambda x: clf(ae.encode(x)), ext_loader, device)
            print(f"=== External Test Accuracy (latent z, {os.path.basename(ext_path)}): {ext_acc:.2f}% ===")
        except Exception as e:
            print(f"[ERROR] Could not evaluate external test CSV: {e}")

if __name__ == "__main__":
    main()

#%% new balanced dataset loading

# balance_macs_positions.py
# ------------------------------------------------------------
# Editor-only helper to balance CSI data across MACs & positions.
# - Handles CSVs with either 2 cols [MAC, CSI] or 4 cols [MAC, RSSI, NOISE, CSI]
# - Valid rows require CSI column to have exactly 128 space-separated numbers
# - Position inferred from filename by regex r'(p\d+)'  (e.g., "...p4_*.csv" -> position "p4")
# - Maximizes K so that each (mac, position) has exactly K rows
# - Returns data in-memory; optionally writes CSVs if you pass out_dir
# ------------------------------------------------------------

import os
import re
import csv
import random
from typing import List, Dict, Tuple, Any
from collections import defaultdict


def extract_position_from_filename(filepath: str) -> str:
    """
    Extract 'pX' (e.g., 'p4') from filename. Returns 'unknown' if not found.
    """
    fname = os.path.basename(filepath)
    m = re.search(r'(p\d+)', fname, flags=re.IGNORECASE)
    return m.group(1).lower() if m else "unknown"


def _get_csi_field(row: List[str]) -> str:
    """
    Given a CSV row that is either [mac, csi] or [mac, rssi, noise, csi],
    return the CSI field or '' if unsupported layout.
    """
    if len(row) == 2:
        return row[1].strip()
    if len(row) == 4:
        return row[3].strip()
    return ""


def _is_valid_csi128(csi: str) -> bool:
    """
    A 'valid' CSI entry has exactly 128 space-separated values.
    """
    parts = csi.split()
    return len(parts) == 128


def parse_valid_rows_by_mac(csv_path: str, target_macids: List[str]) -> Dict[str, List[List[str]]]:
    """
    Return a dict: mac -> list[rows] that are valid and match the target macs.
    - Accepts 2-col or 4-col layout
    - Filters by MAC and CSI length (128)
    """
    keep = defaultdict(list)
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            mac = row[0].strip()
            if mac not in target_macids:
                continue
            csi = _get_csi_field(row)
            if not csi or not _is_valid_csi128(csi):
                continue
            keep[mac].append(row)
    return keep


def compute_uniform_k(available: Dict[str, Dict[str, int]]) -> int:
    """
    Given counts available[mac][pos], return the maximum K such that
    every (mac, pos) has at least K. If any is missing/zero -> K could be 0.
    """
    k = None
    for mac, pos_counts in available.items():
        for pos, cnt in pos_counts.items():
            k = cnt if k is None else min(k, cnt)
    return 0 if k is None else int(k)


def balance_dataset(
    mac_ids: List[str],
    input_csvs: List[str],
    *,
    seed: int = 0,
    shuffle_within_position: bool = True,
    out_dir: str = None,                 # if provided, writes balanced_all.csv and balanced_pX.csv
    return_data: bool = True             # return balanced rows in-memory
) -> Tuple[int, Dict[str, Dict[str, int]], List[List[str]], Dict[str, List[List[str]]]]:
    """
    Create a balanced dataset such that every (mac, position) contributes K rows,
    with K maximized subject to availability across all MACs and positions.

    Args:
        mac_ids:        List of MAC IDs to keep (equalized).
        input_csvs:     List of CSV file paths (typically one per position).
                        Position inferred from filename via r'(p\\d+)'.
        seed:           Random seed for deterministic sampling.
        shuffle_within_position: Shuffle sampled rows before concatenating per position.
        out_dir:        If provided, writes balanced CSVs to this folder.
        return_data:    If True, return the balanced rows in-memory as well.

    Returns:
        K:                          The per-(mac, position) sample count chosen.
        available_counts:           Dict[mac][position] -> original available count.
        balanced_all_rows:          List of all balanced rows (if return_data=True, else []).
        balanced_rows_by_position:  Dict[position] -> list of balanced rows for that position (if return_data=True, else {}).

    Notes:
        - If K == 0, nothing is written and empty lists are returned; available_counts
          still helps you diagnose which (mac, position) was the bottleneck.
        - Row layout is preserved exactly as in the input.
    """
    random.seed(seed)

    # 1) Collect valid rows per (position, mac)
    positions: List[str] = []
    rows_by_pos_mac: Dict[str, Dict[str, List[List[str]]]] = {}
    for path in input_csvs:
        pos = extract_position_from_filename(path)
        positions.append(pos)
        rows_by_pos_mac.setdefault(pos, {})
        rows_by_pos_mac[pos] = parse_valid_rows_by_mac(path, mac_ids)

    # Ensure every (pos, mac) key exists
    for pos in positions:
        for mac in mac_ids:
            rows_by_pos_mac[pos].setdefault(mac, [])

    # 2) Available counts
    available_counts: Dict[str, Dict[str, int]] = {mac: {} for mac in mac_ids}
    for mac in mac_ids:
        for pos in positions:
            available_counts[mac][pos] = len(rows_by_pos_mac[pos][mac])

    # 3) Maximum uniform K
    K = compute_uniform_k(available_counts)

    # 4) If K == 0, early exit with diagnostics
    if K == 0:
        # Optional: print quick hint
        zeros = [(mac, pos) for mac in mac_ids for pos in positions if available_counts[mac][pos] == 0]
        if zeros:
            print("K=0: at least one (mac, position) has zero valid rows.")
            print("First few missing pairs:", zeros[:10], "..." if len(zeros) > 10 else "")
        return 0, available_counts, [], {}

    # 5) Sample K rows for each (mac, pos)
    balanced_all_rows: List[List[str]] = []
    balanced_rows_by_position: Dict[str, List[List[str]]] = {pos: [] for pos in positions}

    for pos in positions:
        for mac in mac_ids:
            pool = rows_by_pos_mac[pos][mac]
            chosen = random.sample(pool, K)   # uniform sample; deterministic via seed
            if shuffle_within_position:
                random.shuffle(chosen)
            balanced_rows_by_position[pos].extend(chosen)
            balanced_all_rows.extend(chosen)

    # 6) Optionally write output CSVs
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        all_path = os.path.join(out_dir, "balanced_all.csv")
        with open(all_path, "w", newline="") as f:
            csv.writer(f).writerows(balanced_all_rows)
        for pos in positions:
            p_path = os.path.join(out_dir, f"balanced_{pos}.csv")
            with open(p_path, "w", newline="") as f:
                csv.writer(f).writerows(balanced_rows_by_position[pos])
        print(f"[balance_dataset] Wrote: {all_path} and {len(positions)} per-position files to {out_dir}")

    # 7) Small summary print
    total_rows = K * len(mac_ids) * len(positions)
    print(f"[balance_dataset] Using K={K} per (mac, position). Total balanced rows = {total_rows}")
    return (K,
            available_counts,
            balanced_all_rows if return_data else [],
            balanced_rows_by_position if return_data else {})


#%%


mac_ids = [
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"00:FC:BA:38:4B:02", \
"6C:B2:AE:39:1A:A0", \
"6C:B2:AE:39:1A:A1", \
"70:0F:6A:DE:EC:A0", \
"70:0F:6A:DE:EC:A1", \
"70:0F:6A:DE:EC:A2", \
#"00:FC:BA:27:63:00", \
#"00:FC:BA:27:63:01", \
#"00:FC:BA:27:63:02"
]
    
csvs = [
    
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\openai_clip_based\individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\p8_second_last.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\openai_clip_based\individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\p7_second_last.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\openai_clip_based\individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\p2_second_last.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\openai_clip_based\individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\p1_second_last.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\openai_clip_based\individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\p3_second_last.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\openai_clip_based\individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\p4_second_last.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\openai_clip_based\individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\p5_second_last.csv", \
    r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\openai_clip_based\individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\p6_second_last.csv", \
]

K, counts, all_rows, by_pos = balance_dataset(
    mac_ids=mac_ids,
    input_csvs=csvs,
    seed=0,
    shuffle_within_position=True,
    out_dir=r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\openai_clip_based\individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\val_balanced",          # or "/path/to/out" if you want CSVs saved
    return_data=True
)

print("Chosen K =", K)
# all_rows is your fully balanced dataset (same number per MAC and per position).
# by_pos['p1'] etc. holds each balanced position subset.

#%% sanity check

# sanity_check_balanced.py
# ------------------------------------------------------------
# Loads balanced CSVs from an output folder and prints, for each file:
#   - counts per (MAC, position)
#   - percentages per position *within each MAC*
# Assumptions match the balancer:
#   - Files: balanced_all.csv and balanced_pX.csv (p1, p2, ...)
#   - CSV rows are either [MAC, CSI] or [MAC, RSSI, NOISE, CSI]
#   - Position inferred from filename ("balanced_p3.csv" -> "p3").
#     For balanced_all.csv, we report totals grouped by position inferred from inputs used to create it
#     (it’s already perfectly balanced, but we still print by position using filenames).
# ------------------------------------------------------------

import os
import re
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable


def _get_csi_field(row: List[str]) -> str:
    if len(row) == 2:
        return row[1].strip()
    if len(row) == 4:
        return row[3].strip()
    return ""


def _is_valid_csi128(csi: str) -> bool:
    return len(csi.split()) == 128


def _infer_position_from_filename(path: str) -> str:
    # For per-position files like balanced_p3.csv -> p3
    fname = os.path.basename(path)
    m = re.search(r'(p\d+)', fname, flags=re.IGNORECASE)
    return m.group(1).lower() if m else "unknown"


def _load_rows(csv_path: str) -> List[List[str]]:
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            csi = _get_csi_field(row)
            if not csi or not _is_valid_csi128(csi):
                continue
            rows.append(row)
    return rows


def _collect_positions_in_folder(folder: str) -> List[str]:
    # look for balanced_pX.csv
    pos = []
    for name in os.listdir(folder):
        if name.startswith("balanced_") and name.endswith(".csv") and name != "balanced_all.csv":
            p = _infer_position_from_filename(name)
            if p not in ("unknown",) and p not in pos:
                pos.append(p)
    pos.sort(key=lambda x: (x[0], int(x[1:])) if re.match(r'p\d+', x) else (x, 0))
    return pos


def _count_by_mac_and_position(rows: Iterable[List[str]], position: str) -> Dict[str, Dict[str, int]]:
    """
    Return counts[mac][position] = count, for a single file that is known to
    represent a single 'position' (e.g., balanced_p3.csv).
    """
    counts = defaultdict(lambda: defaultdict(int))
    for row in rows:
        mac = row[0].strip()
        counts[mac][position] += 1
    return counts


def _merge_counts(dest: Dict[str, Dict[str, int]], src: Dict[str, Dict[str, int]]):
    for mac, perpos in src.items():
        for pos, c in perpos.items():
            dest.setdefault(mac, {})
            dest[mac][pos] = dest[mac].get(pos, 0) + c


def _print_table(counts: Dict[str, Dict[str, int]], positions: List[str], title: str):
    """
    Print a simple grid:
      MAC | p1(count, % of MAC) p2(count, %) ... | Total
    Percentages are per MAC across positions.
    """
    print("\n" + title)
    print("-" * max(60, 20 + 18 * (len(positions) + 1)))
    colw = 18
    header = ["MAC".ljust(colw)] + [p.ljust(colw) for p in positions] + ["Total".ljust(colw)]
    print("".join(header))

    # gather all macs seen (union over positions)
    macs = sorted(counts.keys())
    for mac in macs:
        row = [mac.ljust(colw)]
        total_mac = sum(counts.get(mac, {}).get(p, 0) for p in positions)
        for p in positions:
            c = counts.get(mac, {}).get(p, 0)
            pct = (100.0 * c / total_mac) if total_mac > 0 else 0.0
            cell = f"{c} ({pct:.1f}%)".ljust(colw)
            row.append(cell)
        row.append(str(total_mac).ljust(colw))
        print("".join(row))

    # totals per position (across macs)
    total_row = ["Total".ljust(colw)]
    grand = 0
    for p in positions:
        col_total = sum(counts.get(mac, {}).get(p, 0) for mac in macs)
        total_row.append(str(col_total).ljust(colw))
        grand += col_total
    total_row.append(str(grand).ljust(colw))
    print("".join(total_row))


def sanity_check_balanced_folder(folder: str):
    """
    Scans 'folder' for:
      - balanced_all.csv
      - balanced_p*.csv
    For each per-position file, prints its per-MAC counts (trivially all rows are from that position).
    Then prints a combined summary building (MAC, position) counts across all per-position files,
    showing counts and percentages within each MAC.

    Percentages in brackets are computed *within each MAC* across positions.
    """
    # 1) Find files
    all_path = os.path.join(folder, "balanced_all.csv")
    pos_files = []
    for name in os.listdir(folder):
        if name.startswith("balanced_") and name.endswith(".csv") and name != "balanced_all.csv":
            pos_files.append(os.path.join(folder, name))

    # 2) Collect positions and create a combined count from per-position files
    positions = _collect_positions_in_folder(folder)
    combined_counts = {}  # mac -> pos -> count

    # 3) Per-position tables (quick view)
    for pf in sorted(pos_files):
        pos = _infer_position_from_filename(pf)
        rows = _load_rows(pf)
        cts = _count_by_mac_and_position(rows, pos)
        _print_table(cts, [pos], title=f"Per-file sanity: {os.path.basename(pf)}")
        _merge_counts(combined_counts, cts)

    # 4) Combined per-position summary (across all balanced_pX.csv)
    if positions:
        _print_table(combined_counts, positions, title="Combined summary across all per-position files")

    # 5) Optional: also check balanced_all.csv if present (will match combined if created from same set)
    if os.path.exists(all_path):
        rows_all = _load_rows(all_path)
        # We don’t have per-row position tags inside balanced_all.csv; but since the folder contains
        # balanced_pX.csv, their union equals balanced_all.csv. We already printed combined per-position
        # stats above, which is the main check the dataset is balanced.
        print(f"\n[Info] balanced_all.csv present with {len(rows_all)} valid rows.")
        # A minimal cross-check: verify total matches the combined per-position sum.
        combined_total = sum(sum(perpos.values()) for perpos in combined_counts.values())
        if combined_total == len(rows_all):
            print("[OK] balanced_all.csv total matches the sum of per-position files.")
        else:
            print(f"[WARN] balanced_all.csv total ({len(rows_all)}) != sum of per-position files ({combined_total}).")

#%%


sanity_check_balanced_folder(r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\openai_clip_based\individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\train_balanced")
