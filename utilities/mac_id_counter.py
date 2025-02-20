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
    for mac, count in sorted_mac_counts:
        print(f"{mac}: {count}")
    
    # Print total count of unique MAC addresses
    print("\nTotal Unique MAC IDs:", len(mac_counter))
    print("Total Entries:", total_entries)
    
    # Extract the first n MAC addresses
    top_n_macs = sorted_mac_counts[:n]
    
    # Print the first n MAC addresses in the required format
    print(f"\nTop {n} MAC Addresses:")
    for i, (mac, _) in enumerate(top_n_macs):
        if i == len(top_n_macs) - 1:
            print(f'"{mac}"')  # Last entry without trailing comma
        else:
            print(f'"{mac}", \\')
    
    # Print the count for the least number in these n MAC IDs
    if top_n_macs:
        least_count = top_n_macs[-1][1]
        print(f"\nCount for the least number in these {n} MAC IDs:", least_count)


