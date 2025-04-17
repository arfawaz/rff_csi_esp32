# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 20:57:38 2025

@author: fawaz243
"""


import os
import csv
import re
from collections import defaultdict

# ------------------------- Function 1 -------------------------
def extract_position_from_filename(filename):
    """
    Extract the position identifier (e.g., "p4", "p5", etc.) from a filename.

    Args:
        filename (str): Name of the CSV file (e.g., "p4_data.csv")

    Returns:
        str: The extracted position identifier (e.g., "p4"), or "unknown" if not found
    """
    match = re.search(r'(p\d+)', filename)  # Regex to find "p" followed by digits
    return match.group(1) if match else "unknown"  # Return "p4" etc., or "unknown" if not found


# ------------------------- Function 2 -------------------------
def count_valid_macid_occurrences(csv_path, target_macids):
    """
    Count how many valid entries exist for each target MAC ID in a given CSV file.

    A valid entry is one where:
    - MAC ID is in the target list
    - The CSI data column contains exactly 128 space-separated values

    Args:
        csv_path (str): Full path to the CSV file
        target_macids (list[str]): List of MAC IDs to filter and count

    Returns:
        dict[str, int]: Dictionary mapping each valid MAC ID to its number of valid entries
    """
    counts = defaultdict(int)  # Store counts for each MAC ID

    with open(csv_path, 'r') as file:
        reader = csv.reader(file)

        for row in reader:
            if not row:
                continue  # Skip empty rows

            macid = row[0].strip()  # MAC ID is always the first column

            # Determine which column contains CSI data based on number of columns
            if len(row) == 2:
                csi_str = row[1].strip()
            elif len(row) == 4:
                csi_str = row[3].strip()
            else:
                continue  # Skip rows with unknown format

            # Validate CSI data length
            if len(csi_str.split()) != 128:
                continue  # Skip rows with incomplete CSI

            # Count if MAC ID is one of the targets
            if macid in target_macids:
                counts[macid] += 1

    return counts


# ------------------------- Function 3 -------------------------
def macid_position_table_same_path(mac_ids, file_paths):
    """
    Build and print a table showing how many valid entries each MAC ID has per position.

    Two tables are printed:
    1. Original table: actual counts and percentages
    2. Adjusted table: normalized values where total entries per MAC are scaled to the same min count

    Args:
        mac_ids (list[str]): List of target MAC IDs to count
        file_paths (list[str]): List of full paths to CSV files

    Returns:
        tuple:
            - count_dict (dict): Original counts → mac_id → position → count
            - adjusted_counts (dict): Adjusted counts → mac_id → position → normalized float
    """
    count_dict = {mac: defaultdict(int) for mac in mac_ids}  # Counts per MAC ID and position
    total_mac_counts = defaultdict(int)                      # Total count per MAC ID
    position_counts = defaultdict(int)                       # Total count per position
    positions = set()  # Unique positions seen across all files

    # Iterate through each file and update counts
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        position = extract_position_from_filename(filename)
        positions.add(position)

        file_counts = count_valid_macid_occurrences(file_path, mac_ids)

        for mac in mac_ids:
            count = file_counts.get(mac, 0)
            count_dict[mac][position] += count           # Count for MAC at that position
            total_mac_counts[mac] += count               # Global count for MAC
            position_counts[position] += count           # Column total for that position

    # Sort positions for consistent column order
    sorted_positions = sorted(positions)
    col_width = 18  # Width of each column for pretty printing

    # Calculate minimum total count (used for normalization)
    min_total = min(total_mac_counts.values())

    # ----------------- Print Original Table -----------------
    header = ["MAC ID".ljust(col_width)] + [pos.ljust(col_width) for pos in sorted_positions] + ["Total".ljust(col_width)]
    print("Original Table (Counts & Percentages)".center(len(header) * col_width, "-"))
    print("".join(header))

    for mac in mac_ids:
        row = [mac.ljust(col_width)]
        row_total = 0
        for pos in sorted_positions:
            count = count_dict[mac].get(pos, 0)
            percentage = (count / total_mac_counts[mac] * 100) if total_mac_counts[mac] > 0 else 0
            row.append(f"{count} ({percentage:.1f}%)".ljust(col_width))
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

    # ----------------- Print Adjusted Table -----------------
    print("\nAdjusted Table (Based on Minimum Total)".center(len(header) * col_width, "-"))
    print("".join(header))

    adjusted_counts = defaultdict(dict)

    for mac in mac_ids:
        row = [mac.ljust(col_width)]
        row_total = 0.0
        for pos in sorted_positions:
            count = count_dict[mac].get(pos, 0)
            percentage = (count / total_mac_counts[mac]) if total_mac_counts[mac] > 0 else 0
            adjusted = percentage * min_total  # Normalized count
            adjusted_counts[mac][pos] = adjusted
            row.append(f"{adjusted:.1f}".ljust(col_width))
            row_total += adjusted
        row.append(f"{row_total:.1f}".ljust(col_width))
        print("".join(row))

    # Print adjusted column totals
    column_totals_row = ["Total".ljust(col_width)]
    total_of_totals = 0.0
    for pos in sorted_positions:
        total = sum(adjusted_counts[mac][pos] for mac in mac_ids)
        column_totals_row.append(f"{total:.1f}".ljust(col_width))
        total_of_totals += total
    column_totals_row.append(f"{total_of_totals:.1f}".ljust(col_width))
    print("".join(column_totals_row))

    return count_dict, adjusted_counts
