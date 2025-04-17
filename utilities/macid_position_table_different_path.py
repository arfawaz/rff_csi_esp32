# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 21:32:18 2025

@author: fawaz243
"""

import os
import csv
import re
from collections import defaultdict

# -----------------------------------------------------------------------------
# Overall Purpose of the function:
# The purpose of the function `build_and_adjust_macid_tables` is to calculate 
# the occurrence counts of specific MAC IDs across multiple CSV files representing 
# two different datasets (base and target). It then normalizes the counts from 
# the base set to create an adjusted sample count for the target set.
# It helps to understand the distribution of MAC IDs across different positions 
# (extracted from filenames) and adjust the target set based on the base set 
# distribution.

# Input Parameters:
# mac_ids (List[str]): List of MAC IDs to track and analyze.
# file_paths_percent_base (List[str]): List of file paths for the base dataset 
#                                      (used to compute the reference distribution).
# file_paths_adjust_target (List[str]): List of file paths for the target dataset 
#                                       (adjusted based on base set distribution).
# min_samples (Optional[int]): Minimum sample size for adjusting the target set.
#                               If None, the smallest total count from the target set is used.
#
# Return Value:
# This function returns a dictionary of adjusted counts for the target set, 
# scaled according to the percentages from the base set.
# Example: 
# {
#     'mac_id_1': {'p1': adjusted_count, 'p2': adjusted_count, ...},
#     'mac_id_2': {'p1': adjusted_count, 'p2': adjusted_count, ...},
# }
# -----------------------------------------------------------------------------

def extract_position_from_filename(filename):
    """
    Extract the position (e.g., p1, p2, etc.) from the given filename.
    The position is assumed to be of the form 'pX' where X is a number.
    
    Args:
    - filename (str): The filename from which the position is to be extracted.
    
    Returns:
    - str: Position extracted from the filename (e.g., 'p1'). 
          If no position is found, returns 'unknown'.
    """
    match = re.search(r'(p\d+)', filename)
    return match.group(1) if match else "unknown"

def count_valid_macid_occurrences(csv_path, target_macids):
    """
    Count occurrences of the specified MAC IDs in the given CSV file, only 
    considering rows that have valid CSI data (exactly 128 integers).
    
    Args:
    - csv_path (str): Path to the CSV file to process.
    - target_macids (List[str]): List of MAC IDs to count.
    
    Returns:
    - defaultdict[int]: A dictionary with MAC IDs as keys and their counts as values.
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

def macid_position_table_different_path(mac_ids, file_paths_percent_base, file_paths_adjust_target, min_samples=None):
    """
    Build and adjust MAC ID tables based on counts from base and target datasets.
    It calculates the raw counts of MAC IDs for both the base and target sets, 
    adjusts the counts from the target set based on the percentage distribution 
    from the base set, and prints the results in a formatted table.
    
    Args:
    - mac_ids (List[str]): List of MAC IDs to analyze.
    - file_paths_percent_base (List[str]): List of file paths for the base dataset 
                                             (used for calculating reference distribution).
    - file_paths_adjust_target (List[str]): List of file paths for the target dataset 
                                             (adjusted based on base set distribution).
    - min_samples (Optional[int]): Minimum sample count for the target set.
                                    If None, the minimum total from the target set is used.
    
    Returns:
    - dict: A dictionary containing the adjusted counts for each MAC ID in the target set,
            per position (scaled to match the base distribution).
    """
    
    col_width = 18  # Column width for formatting tables

    # Function to build counts from a set of files (either base or target)
    def build_counts(file_paths, label):
        """
        Count MAC ID occurrences in the given file paths, grouping by position 
        (extracted from filenames) and calculating percentages of occurrences 
        per MAC ID.
        
        Args:
        - file_paths (List[str]): List of file paths to process.
        - label (str): Label to indicate whether it's the base or target set.
        
        Returns:
        - tuple: (count_dict, total_mac_counts, sorted_positions)
          - count_dict (dict): MAC ID counts per position.
          - total_mac_counts (dict): Total counts for each MAC ID across all positions.
          - sorted_positions (List[str]): Sorted list of positions (extracted from filenames).
        """
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

        # Sort positions for display purposes
        sorted_positions = sorted(positions)

        # Print the counts and percentages table
        print(f"\n{label} Table (Original Counts and Percentages)".center(len(sorted_positions) * col_width + 3 * col_width, "-"))
        header = ["MAC ID".ljust(col_width)] + [pos.ljust(col_width) for pos in sorted_positions] + ["Total".ljust(col_width)]
        print("".join(header))

        # Print MAC ID rows with their counts and percentages
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

        # Print totals for each position
        total_row = ["Total".ljust(col_width)]
        grand_total = 0
        for pos in sorted_positions:
            total = position_counts[pos]
            total_row.append(str(total).ljust(col_width))
            grand_total += total
        total_row.append(str(grand_total).ljust(col_width))
        print("".join(total_row))

        return count_dict, total_mac_counts, sorted_positions

    # Build counts for the base set (used to calculate reference percentages)
    base_counts, base_total_mac_counts, base_positions = build_counts(file_paths_percent_base, "BASE SET")

    # Calculate the percentage distribution per MAC ID in the base set
    base_percentages = defaultdict(dict)
    for mac in mac_ids:
        total = base_total_mac_counts[mac]
        for pos in base_positions:
            count = base_counts[mac][pos]
            base_percentages[mac][pos] = (count / total * 100) if total > 0 else 0

    # Build counts for the target set (used for adjustment)
    target_counts, target_total_mac_counts, target_positions = build_counts(file_paths_adjust_target, "TARGET SET")

    # Determine the minimum total count across all MAC IDs in the target set
    if min_samples is None:
        min_total_target = min(target_total_mac_counts.values())
    else:
        min_total_target = min_samples

    # Adjust the target set based on the percentage distribution from the base set
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

    # Print the final total row for adjusted counts
    total_row = ["Total".ljust(col_width)]
    grand_total = 0
    for pos in target_positions:
        total = sum(adjusted_target_counts[mac][pos] for mac in mac_ids)
        total_row.append(f"{total:.1f}".ljust(col_width))
        grand_total += total
    total_row.append(f"{grand_total:.1f}".ljust(col_width))
    print("".join(total_row))

    return adjusted_target_counts
