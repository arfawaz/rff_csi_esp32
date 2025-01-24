
# PYTHON IMPORTS
import csv
import torch
import csv  # Module for working with CSV files
import torch  # PyTorch library for tensor operations

'''
This code processes a CSV file containing CSI (Channel State Information) data and MAC addresses. It does the following:

1) Parses CSI Data: Each row in the CSV contains a MAC address and corresponding CSI data. The CSI data is parsed into a PyTorch tensor of shape (64, 2), where each row represents a subcarrier's magnitude and angle.

2) Assigns Labels to MAC Addresses: Each unique MAC address is assigned an integer label. These labels are stored in a dictionary to ensure consistency across samples.

3) Limits Samples per MAC Address: The function ensures that no more than a specified maximum number of samples (max_samples_per_mac) are collected for each MAC address.

4) Outputs Data: After processing the entire CSV file:

5) The processed CSI data is returned as a stacked PyTorch tensor (data_).
6) The corresponding MAC address labels are returned as another tensor (labels_).
'''

# Function to parse each row and process it into a 64x2 tensor for CSI data
def parse_csi_data(csi_row):
    """
    Parses a single row of CSI data into a 64x2 PyTorch tensor.
    
    Args:
    - csi_row (str): A string representing the CSI row data, consisting of alternating magnitude and angle values.

    Returns:
    - torch.Tensor: A 64x2 tensor containing magnitude and angle pairs.
    - None: If the data is invalid (e.g., does not have 128 values or contains non-numeric values).
    """
    # Split the CSI row into individual values based on spaces
    csi_values = csi_row.split()
    
    # Check if the row contains exactly 128 values (64 subcarriers * 2 values each: magnitude and angle)
    if len(csi_values) != 128:
        return None  # Return None for invalid rows
    
    # Create a 64x2 tensor to store magnitude and angle pairs
    csi_tensor = []
    
    # Iterate through the CSI values two at a time (magnitude and angle)
    for i in range(0, 128, 2):  # Step size of 2
        try:
            # Convert the magnitude and angle to floats
            magnitude = float(csi_values[i])
            angle = float(csi_values[i + 1])
            # Append the pair as a list to the tensor
            csi_tensor.append([magnitude, angle])
        except ValueError:
            # If a value cannot be converted to a float, return None (invalid row)
            return None
    
    # Convert the list of magnitude and angle pairs to a PyTorch tensor and return it
    return torch.tensor(csi_tensor)

# Function to process the entire CSV file
def process_csv(file_path, max_samples_per_mac=9000):
    """
    Processes a CSV file to extract CSI data, convert it to tensors, and assign integer labels to each MAC address.
    
    Args:
    - file_path (str): The path to the CSV file.
    - max_samples_per_mac (int): The maximum number of samples to extract for each unique MAC address.

    Returns:
    - torch.Tensor: A tensor containing all the processed 64x2 CSI tensors.
    - torch.Tensor: A tensor containing integer labels corresponding to each MAC address.
    """
    data = []   # List to store the processed 64x2 tensors
    labels = [] # List to store the integer labels for each tensor
    
    mac_id_to_label = {}  # Dictionary to map each unique MAC address to an integer label
    mac_id_sample_count = {}  # Dictionary to track the number of samples collected for each MAC address
    label_counter = 0  # Counter to assign unique integer labels to MAC addresses
    
    # Open the CSV file in read mode
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)  # Create a CSV reader object
        
        mac_id = None  # Placeholder for the current MAC ID
        for row in reader:  # Iterate through each row in the CSV file
            if len(row) != 2:
                continue  # Skip rows that do not have exactly 2 columns (MAC ID and CSI data)
            
            current_mac_id, csi_row = row  # Unpack the row into MAC ID and CSI data
            
            # Parse the CSI data row into a 64x2 tensor
            csi_tensor = parse_csi_data(csi_row)
            if csi_tensor is not None:  # If the row is valid and successfully parsed
                # Assign a new integer label to the MAC address if it is encountered for the first time
                if current_mac_id not in mac_id_to_label:
                    mac_id_to_label[current_mac_id] = label_counter
                    label_counter += 1  # Increment the label counter for the next unique MAC address
                
                # Retrieve the integer label for the current MAC address
                label = mac_id_to_label[current_mac_id]
                
                # Initialize the sample count for the MAC ID if not already present
                if current_mac_id not in mac_id_sample_count:
                    mac_id_sample_count[current_mac_id] = 0
                
                # Add the sample if the current MAC ID has not yet reached the maximum allowed samples
                if mac_id_sample_count[current_mac_id] < max_samples_per_mac:
                    data.append(csi_tensor)  # Append the 64x2 tensor to the data list
                    labels.append(label)  # Append the corresponding integer label
                    mac_id_sample_count[current_mac_id] += 1  # Increment the sample count for this MAC ID
                
                # Skip further processing if the sample count for this MAC ID reaches the maximum
                if mac_id_sample_count[current_mac_id] >= max_samples_per_mac:
                    continue

    # Stack all the 64x2 tensors into a single PyTorch tensor
    data_ = torch.stack(data)
    # Convert the list of labels into a PyTorch tensor
    labels_ = torch.tensor(labels, dtype=torch.long)
    return data_, labels_
