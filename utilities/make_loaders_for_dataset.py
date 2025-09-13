# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 17:16:39 2025

@author: fawaz243
"""

from torch.utils.data import DataLoader, Subset

def make_loaders_for_dataset(dataset, tr_idx, va_idx, batch_size=64, num_workers=0):
    """
    Construct **train** and **validation** DataLoaders from a single base `dataset`
    using **precomputed index lists** (e.g., from a stratified split).

    Why this is useful
    ------------------
    - You often create one canonical dataset (e.g., all CSI samples with labels),
      then split it into TRAIN/VAL by **indices** so different models (CNN, CLIP)
      can consume **the exact same samples** for fair comparison.
    - Wrapping the base dataset with `torch.utils.data.Subset` ensures that the
      underlying items (and their order, when `shuffle=False`) are identical
      across pipelines.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The full dataset to slice. Each __getitem__(i) should return a sample
        (e.g., `(x, y)`), where `i` refers to the original global index.
    tr_idx : Sequence[int]
        Indices for the TRAIN split (e.g., from `stratified_train_val_indices`).
    va_idx : Sequence[int]
        Indices for the VAL split.
    batch_size : int, default=64
        Mini-batch size for both train and val loaders. (Use the same by default
        for simplicity; you can change later if desired.)
    num_workers : int, default=0
        Number of background worker processes for data loading. Set >0 to
        parallelize CPU-side preprocessing/IO; 0 is simplest & most portable.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        DataLoader over the TRAIN subset.
        - `shuffle=True` to randomize sample order each epoch (improves SGD).
        - `drop_last=True` so every batch has identical size; this is important
          for **contrastive losses**, **BatchNorm**, and when aggregating across
          GPUs, where variable last-batches can cause shape mismatches.
        - `pin_memory=True` for faster host→GPU transfer when using CUDA.
    val_loader : torch.utils.data.DataLoader
        DataLoader over the VAL subset.
        - `shuffle=False` to keep evaluation deterministic and aligned with
          saved indices (useful for debugging and exact reproducibility).
        - `drop_last=False` so you evaluate on **all** validation samples.

    Notes
    -----
    - Reproducibility depends on how `tr_idx`/`va_idx` were created (e.g., fixed
      RNG seed). With `shuffle=True`, the *epoch-to-epoch* order within TRAIN
      is randomized; that’s expected and desirable for training.
    - If you need deterministic train ordering too (rare), pass a fixed
      `generator` to the DataLoader and disable shuffling—usually **not** advised.
    """
    # Slice the base dataset into two views backed by the exact same storage.
    ds_train = Subset(dataset, tr_idx)
    ds_val   = Subset(dataset, va_idx)

    # TRAIN loader: shuffle for SGD; drop incomplete final batch for consistent shapes.
    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # VAL loader: no shuffle; keep all samples for metrics.
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
