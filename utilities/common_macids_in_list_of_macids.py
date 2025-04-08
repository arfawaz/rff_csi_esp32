#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 16:14:23 2025

@author: fawaz
"""

def common_macids_in_list_of_macids(mac_lists):
    """
    Takes a list of lists containing MAC IDs and returns a list of MAC IDs
    that are common across all sublists.

    Parameters:
    mac_lists (list of list of str): A list containing multiple lists of MAC ID strings.

    Returns:
    list of str: MAC IDs that appear in every list.
    """
    if not mac_lists:
        return []

    # Start with the set of MACs from the first list
    common_macids = set(mac_lists[0])

    # Intersect with MACs from each subsequent list
    for mac_list in mac_lists[1:]:
        common_macids.intersection_update(mac_list)

    # Return as a list (sorted optionally)
    print(f"Common MAC IDs accross the list of MAC IDs: {common_macids}")
    return list(common_macids)
