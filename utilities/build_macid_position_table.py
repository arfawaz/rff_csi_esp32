# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 16:58:51 2025

@author: fawaz243
"""

import os
import csv
import re
from collections import defaultdict

# =====================================================
# Description:
# This script processes multiple CSV files that contain MAC IDs and their corresponding CSI data.
# It counts how many times each MAC ID appears with valid CSI data (i.e., 128 elements) per file.
# It groups files by "position" (extracted from filenames using a regex, e.g., 'p1', 'p2', etc.).
# The script produces:
#   1. A table showing the raw count and percentage of occurrences of each MAC ID at each position.
#   2. An "adjusted" table where each MAC's distribution is scaled to match the MAC with the lowest total count,
#      to allow fair comparison across MAC IDs.
#
# Inputs:
#   - A list of MAC IDs to track
#   - A list of file paths pointing to CSVs containing CSI data
#
# Output:
#   - Printed tables (original and adjusted) summarizing count statistics
#   - Two dictionaries:
#       - count_dict: raw counts of MAC IDs per position
#       - adjusted_counts: scaled values per position based on the MAC with the lowest total
# =====================================================

def extract_position_from_filename(filename):
    """
    Extracts the position identifier from the filename using a regular expression.
    Example: 'foo_p3_bar.csv' -> 'p3'

    Args:
        filename (str): The name of the CSV file.

    Returns:
        str: The position string (e.g., 'p1', 'p2') or 'unknown' if not found.
    """
    match = re.search(r'(p\d+)', filename)
    return match.group(1) if match else "unknown"

def count_valid_macid_occurrences(csv_path, target_macids):
    """
    Counts valid CSI entries (those with 128 values) for the specified MAC IDs in a CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        target_macids (set): Set of MAC IDs to track.

    Returns:
        dict: Dictionary of counts for each target MAC ID found in the file.
    """
    counts = defaultdict(int)
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue  # skip empty rows

            macid = row[0].strip()

            # Extract CSI string depending on whether there are 2 or 4 columns
            if len(row) == 2:
                csi_str = row[1].strip()
            elif len(row) == 4:
                csi_str = row[3].strip()
            else:
                continue  # Skip malformed rows

            # Check if CSI string contains exactly 128 space-separated elements
            if len(csi_str.split()) != 128:
                continue

            # If the MAC ID is one of the targets, increment its count
            if macid in target_macids:
                counts[macid] += 1
    return counts

def build_macid_position_table(mac_ids, file_paths):
    """
    Builds and prints a table of MAC ID counts across different positions.

    Args:
        mac_ids (list): List of MAC IDs to track.
        file_paths (list): List of CSV file paths.

    Returns:
        tuple:
            - count_dict: dict of {mac_id: {position: count}}
            - adjusted_counts: dict of {mac_id: {position: scaled_value}}
    """
    # Initialize data structures
    count_dict = {mac: defaultdict(int) for mac in mac_ids}
    total_mac_counts = defaultdict(int)
    position_counts = defaultdict(int)  # Total counts per position (for column totals)
    positions = set()

    # Count MAC ID occurrences across files and positions
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
    col_width = 18  # Width of each column in the table

    # Find the MAC ID with the minimum total count for normalization
    min_total = min(total_mac_counts.values())

    # ------------------ Print Original Table ------------------
    header = ["MAC ID".ljust(col_width)] + [pos.ljust(col_width) for pos in sorted_positions] + ["Total".ljust(col_width)]
    print("Original Table (Counts & Percentages)".center(len(header) * col_width, "-"))
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

    # ------------------ Print Adjusted Table ------------------
    print("\nAdjusted Table (Based on Minimum Total)".center(len(header) * col_width, "-"))
    print("".join(header))

    # Adjust values proportionally to match min_total across MAC IDs
    adjusted_counts = defaultdict(dict)  # Store adjusted values
    for mac in mac_ids:
        row = [mac.ljust(col_width)]
        row_total = 0
        for pos in sorted_positions:
            count = count_dict[mac].get(pos, 0)
            percentage = (count / total_mac_counts[mac] * 100) if total_mac_counts[mac] > 0 else 0
            adjusted_value = (percentage / 100) * min_total
            adjusted_counts[mac][pos] = adjusted_value
            row_total += adjusted_value
            row.append(f"{adjusted_value:.1f}".ljust(col_width))
        row.append(f"{row_total:.1f}".ljust(col_width))
        print("".join(row))

    # Print adjusted column totals
    column_totals_row = ["Total".ljust(col_width)]
    total_of_totals = 0
    for pos in sorted_positions:
        total = sum(adjusted_counts[mac][pos] for mac in mac_ids)
        column_totals_row.append(f"{total:.1f}".ljust(col_width))
        total_of_totals += total
    column_totals_row.append(f"{total_of_totals:.1f}".ljust(col_width))
    print("".join(column_totals_row))

    # Return raw and adjusted counts
    return count_dict, adjusted_counts
