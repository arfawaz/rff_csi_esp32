# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 21:54:32 2025

@author: fawaz243
"""

# ---------- process_csv_fixed_id_uniform_sampling_rssi_pos.py ----------
import csv, random, torch
from collections import defaultdict

def parse_csi_data(csi_row: str):
    vals = csi_row.split()
    if len(vals) != 128:
        return None
    out = []
    for i in range(0, 128, 2):
        try:
            mag = float(vals[i]); ang = float(vals[i+1])
        except ValueError:
            return None
        out.append([mag, ang])
    return torch.tensor(out)  # [64,2]

def process_csv_fixed_id_uniform_sampling_rssi_pos(
    file_path: str,
    mac_id_list: list,
    pos_tag_to_id: dict = None,  # e.g., {'p1':0,...,'p8':7}
    max_samples_per_mac: int = 50_000,
    stratify_by_pos: bool = False,
    seed: int = 20250910,
):
    """
    Reads combined CSV with 3 or 5 columns:
      - 3: [MAC, CSI, POS]
      - 5: [MAC, RSSI, NOISE, CSI, POS]
    Returns:
      data:       [N, 64, 2] float32
      mac_labels: [N] long (0..C-1, order of mac_id_list)
      pos_labels: [N] long (per pos_tag_to_id)
    """
    random.seed(seed)
    mac_to_label = {m: i for i, m in enumerate(mac_id_list)}

    # Position mapping (stable across train/test)
    if pos_tag_to_id is None:
        pos_tag_to_id = {f"p{i}": i-1 for i in range(1, 9)}  # p1->0,...,p8->7

    # Collect per-MAC (and optionally per position) entries
    store = defaultdict(list)  # key depends on stratify_by_pos
    with open(file_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) not in (3, 5):
                continue
            if len(row) == 3:
                mac, csi_row, pos = row
            else:  # 5
                mac, _, _, csi_row, pos = row

            if mac not in mac_to_label:
                continue
            pos_lc = pos.strip().lower()
            if pos_lc not in pos_tag_to_id:
                # tolerate names like 'p01'
                pl = pos_lc.lstrip("p")
                if pl.isdigit():
                    pos_lc = f"p{int(pl)}"
                if pos_lc not in pos_tag_to_id:
                    continue  # unknown position tag

            csi_t = parse_csi_data(csi_row)
            if csi_t is None:
                continue

            mac_id = mac_to_label[mac]
            pos_id = pos_tag_to_id[pos_lc]
            if stratify_by_pos:
                key = (mac_id, pos_id)
            else:
                key = mac_id
            store[key].append((csi_t, mac_id, pos_id))

    # Uniform random selection up to max per MAC (or per MAC×POS if stratify)
    data, mac_labels, pos_labels = [], [], []
    if stratify_by_pos:
        # cap per (mac,pos), but also ensure per-MAC cap overall
        by_mac_total = defaultdict(int)
        for (mac_id, pos_id), entries in store.items():
            # available budget left for this MAC
            budget = max_samples_per_mac - by_mac_total[mac_id]
            if budget <= 0:
                continue
            take = min(len(entries), budget)
            for (csi_t, m, p) in random.sample(entries, take):
                data.append(csi_t)
                mac_labels.append(m)
                pos_labels.append(p)
                by_mac_total[mac_id] += 1
    else:
        for mac_id, entries in store.items():
            take = min(len(entries), max_samples_per_mac)
            for (csi_t, m, p) in random.sample(entries, take):
                data.append(csi_t); mac_labels.append(m); pos_labels.append(p)

    if not data:
        return None, None, None

    data = torch.stack(data).squeeze().float()               # [N,64,2]
    mac_labels = torch.tensor(mac_labels, dtype=torch.long)  # [N]
    pos_labels = torch.tensor(pos_labels, dtype=torch.long)  # [N]
    return data, mac_labels, pos_labels
