#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:17:42 2025

@author: fawaz
"""

import numpy as np  # Import NumPy for numerical computations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
from scipy.fft import fft, ifft  # Import FFT and IFFT for signal processing
import csv  # Import CSV module to read the CSV file

def low_pass_filter(signal, cutoff_freq=0.1):
    """
    Applies a low-pass filter using FFT by zeroing out high-frequency components.
    
    :param signal: 1D NumPy array (time-domain signal)
    :param cutoff_freq: Normalized cutoff frequency (0 to 1), where 1 represents Nyquist frequency.
    :return: Filtered signal in the time domain
    """
    fft_coeffs = fft(signal)  # Compute the Fast Fourier Transform (FFT) to convert the signal to frequency domain
    num_coeffs = len(fft_coeffs)  # Get the total number of coefficients (frequency components)
    
    cutoff_index = int(cutoff_freq * num_coeffs)  # Determine the index to cut off high frequencies
    fft_coeffs[cutoff_index:-cutoff_index] = 0  # Set high-frequency components to zero to filter them out
    
    return np.real(ifft(fft_coeffs))  # Apply Inverse FFT (IFFT) to transform back to time domain and keep only real part

def parse_csi_amp_phase_fft_plot(file_path, target_macs, subcarriers, cutoff_freq=0.1, start_time=0, end_time=None, plot_raw=True, plot_fil=True, plot_phase=True):
    """
    Parses the CSI CSV file, extracts data for specific MAC addresses, and processes amplitude and phase.
    It also applies an FFT-based low-pass filter and plots the results.

    :param file_path: Path to the CSV file containing CSI data
    :param target_macs: List of MAC addresses to filter
    :param subcarriers: List of subcarrier indices (1 to 64) to analyze
    :param cutoff_freq: Cutoff frequency for the low-pass filter
    :param start_time: Start index for plotting (time-domain sample index)
    :param end_time: End index for plotting (None means full range)
    :param plot_raw: Boolean flag to plot raw amplitude data
    :param plot_fil: Boolean flag to plot filtered amplitude data
    :param plot_phase: Boolean flag to plot filtered phase data
    """
    
    # Dictionaries to store amplitude, phase, and timestamps for each MAC address
    data_amp = {mac: [] for mac in target_macs}  # Amplitude data storage
    data_phase = {mac: [] for mac in target_macs}  # Phase data storage
    timestamps = {mac: [] for mac in target_macs}  # Time indices storage

    # Open and read the CSV file
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            if len(line) < 2:
                continue  # Skip invalid rows (e.g., empty lines)
            
            mac = line[0].strip()  # Extract and clean MAC address from the row
            if mac in target_macs:  # Only process data if the MAC address is in the target list
                csi_values = list(map(int, line[1].split()))  # Convert CSI values from string to integer list
                
                # Extract real and imaginary parts from alternating indices
                imag_parts = np.array(csi_values[0::2])  # Odd indices correspond to imaginary parts
                real_parts = np.array(csi_values[1::2])  # Even indices correspond to real parts
                
                # Compute amplitude and phase for the CSI data
                amplitudes = np.sqrt(real_parts**2 + imag_parts**2)  # Compute magnitude
                phases = np.arctan2(imag_parts, real_parts)  # Compute phase angle using arctan2
                
                # Select only the specified subcarriers
                selected_amplitudes = amplitudes[np.array(subcarriers) - 1]
                selected_phases = phases[np.array(subcarriers) - 1]
                
                # Append the extracted data to respective dictionaries
                data_amp[mac].append(selected_amplitudes)
                data_phase[mac].append(selected_phases)
                timestamps[mac].append(len(data_amp[mac]))  # Use row index as time reference

    # === Plot Amplitude Data ===
    plt.figure(figsize=(12, 6))  # Create a new figure for amplitude plot
    color_map = plt.cm.get_cmap("tab10", len(target_macs) * len(subcarriers))  # Generate a color map
    color_idx = 0  # Initialize color index

    for mac in target_macs:
        if not data_amp[mac]:
            print(f"No data found for MAC: {mac}")
            continue  # Skip if no data available
        
        raw_data = np.array(data_amp[mac])  # Convert amplitude list to NumPy array
        filtered_data = np.apply_along_axis(low_pass_filter, axis=0, arr=raw_data, cutoff_freq=cutoff_freq)  # Apply filtering
        
        if end_time is None:
            end_time = len(filtered_data)  # Use full range if not specified
        
        if start_time < 0 or end_time > len(filtered_data) or start_time >= end_time:
            print(f"Invalid time range for {mac}: {start_time} to {end_time}")
            continue  # Validate time range

        for i, subcarrier in enumerate(subcarriers):
            unique_color = color_map(color_idx)  # Assign color
            color_idx += 1
            
            if plot_raw:
                plt.plot(timestamps[mac][start_time:end_time], 
                         raw_data[start_time:end_time, i], 
                         linestyle='solid', alpha=0.5, color=unique_color, 
                         label=f'Raw {mac} SC {subcarrier}')
            
            if plot_fil:
                plt.plot(timestamps[mac][start_time:end_time], 
                         filtered_data[start_time:end_time, i], 
                         linestyle='dashed', color=unique_color, 
                         label=f'Filtered {mac} SC {subcarrier}')

    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title("CSI Amplitude vs. Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    # === Plot Phase Data ===
    if plot_phase:
        plt.figure(figsize=(12, 6))  # Create a new figure for phase plot
        color_idx = 0  # Reset color index

        for mac in target_macs:
            if not data_phase[mac]:
                continue  # Skip if no data
            
            raw_phase = np.array(data_phase[mac])  # Convert phase list to NumPy array
            filtered_phase = np.apply_along_axis(low_pass_filter, axis=0, arr=raw_phase, cutoff_freq=cutoff_freq)  # Apply filtering

            for i, subcarrier in enumerate(subcarriers):
                unique_color = color_map(color_idx)
                color_idx += 1
                
                if plot_fil:
                    plt.plot(timestamps[mac][start_time:end_time], 
                             filtered_phase[start_time:end_time, i], 
                             linestyle='dashed', color=unique_color, 
                             label=f'Filtered Phase {mac} SC {subcarrier}')
        
        plt.xlabel("Time (samples)")
        plt.ylabel("Phase (radians)")
        plt.title("CSI Phase vs. Time")
        plt.legend()
        plt.grid(True)
        plt.show()

file_path = "/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/esp_printed_antenna/04_feb_2025/msc_allaps_fixedpos_16_feb_3.csv"
#target_macs = ["34:5F:45:A9:A4:19"]

target_macs = ["00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"00:FC:BA:38:4B:02"]
   
#target_macs = ["34:5F:45:A9:A4:19" , "34:5F:45:A8:3C:19", "20:43:A8:64:3A:C1"]  # List of MAC addresses
selected_subcarriers = [40]
start_sample = 1000
end_sample = 1100

parse_csi_amp_phase_fft_plot(
    file_path=file_path,
    target_macs=target_macs,
    subcarriers=selected_subcarriers,
    cutoff_freq=0.1,
    start_time=start_sample,
    end_time=end_sample,
    plot_raw=True,
    plot_fil=True,
    plot_phase=False
)
