# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 17:05:26 2025

@author: fawaz243
"""

def stratified_train_val_indices(y_tensor, train_ratio=0.9, seed=SEED):
    """
    Build a **deterministic, stratified** split of indices into TRAIN and VAL sets.

    What this does
    --------------
    - Takes a 1-D tensor of integer class labels (e.g., MAC IDs mapped to 0..C-1).
    - Preserves the **per-class distribution** when splitting into train/val:
      each class c is split independently in proportion to `train_ratio`.
    - Ensures that for any class with more than one sample, **both** splits
      receive at least one example (when possible).
    - Returns **lists of indices** (into your original dataset) for train and val.
    - The split is **reproducible** because shuffling per class is driven by `seed`.

    Parameters
    ----------
    y_tensor : torch.Tensor
        1-D tensor of length N with integer labels in [0, C-1].
    train_ratio : float, default=0.9
        Fraction of each class to assign to the TRAIN split. The remainder
        goes to VAL. Must be in (0, 1). (Not explicitly checked here.)
    seed : int
        Seed for Python's `random.Random` used to shuffle indices **within each class**.

    Returns
    -------
    train_idx : list[int]
        Sorted list of dataset indices assigned to TRAIN.
    val_idx   : list[int]
        Sorted list of dataset indices assigned to VAL.

    Notes
    -----
    - Classes with a **single** sample (n == 1) are assigned entirely to TRAIN
      (VAL gets 0 for that class). This avoids empty TRAIN for that class.
    - Sorting the final index lists provides a stable ordering (useful if you
      later wrap them in `Subset` and want deterministic iteration when
      `shuffle=False`).
    - Time complexity is O(N log N) mainly due to final sorting; memory is O(N).
    """
    # Create an independent RNG so global seeding elsewhere doesn’t affect this split.
    rng = random.Random(seed)

    # Group dataset indices by class label: by_class[c] = [idx0, idx1, ...]
    by_class = defaultdict(list)
    for idx, yy in enumerate(y_tensor.tolist()):
        by_class[int(yy)].append(idx)

    train_idx, val_idx = [], []

    # Split **per class** to preserve class proportions in each split.
    for c, idxs in by_class.items():
        # Shuffle indices within this class so the split is random but reproducible (via seed).
        rng.shuffle(idxs)

        n = len(idxs)
        # Nominal count for train for this class, rounded to nearest integer.
        n_train = int(round(n * train_ratio))

        # If the class has >1 sample, enforce at least one sample in BOTH splits:
        #   - lower bound: at least 1 goes to TRAIN
        #   - upper bound: at least 1 remains for VAL
        if n > 1:
            n_train = min(max(n_train, 1), n - 1)

        # First n_train indices → TRAIN; remainder → VAL
        train_idx += idxs[:n_train]
        val_idx   += idxs[n_train:]

    # Sort for stable, deterministic ordering (useful when creating DataLoaders with shuffle=False)
    train_idx.sort()
    val_idx.sort()

    return train_idx, val_idx
