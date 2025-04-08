#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 19:12:45 2025

@author: fawaz
"""


import csv
from collections import Counter


    
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
            print(f"\nTotal number of common MAC IDs is less than the requested top count. \
                  Total Requested = {number_of_top_mac_ids} \
                  Actual available = {len(common_mac_ids)}")
        
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
        
        print(f"Total Requested = {number_of_top_mac_ids} \
        Actual available = {len(common_mac_ids)}")
        
        # Find MAC ID with the least occurrences in the top list
        least_mac, least_count = min(top_mac_ids, key=lambda x: x[1], default=(None, None))
        print(f"\nMAC ID with the least occurrences in the top list: {least_mac} ({least_count} samples)")
    
    # Case 2: When minimum_number_of_samples is specified
    else:
        filtered_mac_ids = [(mac, count) for mac, count in sorted_common_mac_counts if count >= minimum_number_of_samples]
        
        if len(filtered_mac_ids) < number_of_top_mac_ids:
            print(f"\nThe required number_of_top_mac_ids with minimum_number_of_samples is not met. \
                  Total Requested = {number_of_top_mac_ids} \
                  Actual available = {len(common_mac_ids)}")
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
        


