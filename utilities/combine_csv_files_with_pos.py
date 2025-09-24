# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 21:21:38 2025

@author: fawaz243
"""

# ---------- combine_csv_files_with_pos.py ----------
import os, re
import pandas as pd

def _infer_pos_tag(path: str):
    """
    Returns 'p1'..'p99' if found in filename or parent dir.
    Matches 'sampled_p1.csv', '/.../p1/file.csv', etc.
    """
    s = path.replace("\\", "/").lower()
    m = re.search(r"(?:^|/)(p\d{1,2})(?:/|$)|sampled_(p\d{1,2})", s)
    if not m:
        raise ValueError(f"Cannot infer position tag from path: {path}")
    return (m.group(1) or m.group(2)).lower()  # 'p1', 'p2', ...

def combine_csv_files_with_pos(file_paths, output_file):
    """
    Combine CSVs (each with 2 or 4 cols) and append POS tag inferred from filename/dir.
    Output columns (no header): [MAC, RSSI, NOISE, CSI, POS]
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    frames = []
    for fp in file_paths:
        pos = _infer_pos_tag(fp)  # 'p1'..'p8'
        try:
            df = pd.read_csv(fp, header=None)
        except Exception as e:
            print(f"Skip {fp}: {e}")
            continue

        if df.shape[1] == 2:
            # [MAC, CSI] -> pad RSSI/NOISE
            df = df.iloc[:, :2]
            df.columns = ["mac", "csi"]
            df["rssi"] = ""
            df["noise"] = ""
            df = df[["mac", "rssi", "noise", "csi"]]
        elif df.shape[1] == 4:
            # [MAC, RSSI, NOISE, CSI]
            df = df.iloc[:, :4]
            df.columns = ["mac", "rssi", "noise", "csi"]
        else:
            print(f"Skipping {fp}: unexpected columns = {df.shape[1]}")
            continue

        df["pos"] = pos  # append position tag
        frames.append(df)

    if not frames:
        raise RuntimeError("No valid CSVs to combine.")

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(output_file, index=False, header=False)
    print(f"Combined CSV with POS saved to: {output_file}")
