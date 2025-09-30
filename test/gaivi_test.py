#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 12:39:49 2025

@author: fawaz
"""

# --- make project_root importable everywhere (terminal/editor/new machine) ---
# ---- make project_root importable (works in Spyder/terminal/any CWD) ----
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent  # folder that contains /utilities and /models
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# -------------------------------------------------------------------------

from utilities.csi_dataset_creator_fixed_id import process_csv_fixed_id
from utilities.csi_dataset_creator_fixed_id_uniform_sampling import process_csv_fixed_id_uniform_sampling
from utilities.process_csv_fixed_id_uniform_smapling_rssi import process_csv_fixed_id_uniform_sampling_rssi
from utilities.csi_dataset_creator import process_csv
from utilities.mean_norm import mean_norm
from utilities.train_test import train, test
from utilities.train_test_loader import train_test_loader
from utilities.train_vit_model_2 import train_vit_model_2
from utilities.test_vit_model_2 import test_vit_model_2
from utilities.CustomDataset_vit_model_2 import CustomDataset_vit_model_2
from models.models import SimpleCNN, vit_model_2, ResNet50CSI, CSIEncoder, LabelEmbedder, CSIResNet50Encoder, CSI_CLIP, CNN_2, LabelHexProjector, LabelHexPlusLoc
from torch.utils.data import Subset, DataLoader, TensorDataset, random_split
from transformers import AdamW
import torch.nn as nn
import torch
import torch.optim as optim
from utilities.losses import clip_loss
from utilities.clip_dataset import CSIDataset
from utilities.evaluate_zero_shot import evaluate_zero_shot
from utilities.train_test import train_clip
from utilities.stratified_train_val_indices import stratified_train_val_indices
from utilities.make_loaders_for_dataset import make_loaders_for_dataset
import os, random
from collections import defaultdict
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights
from utilities.set_global_seed import set_global_seed
from utilities.losses import clip_loss
from utilities.evaluate_zero_shot import evaluate_zero_shot
import json, datetime
from pathlib import Path
import optuna
from optuna.pruners import MedianPruner
from torch.utils.data import DataLoader

# from utilities.optuna_helper_functions import (
#     _jsonify,
#     save_study_best,
#     save_checkpoint,
#     HAS_DATASETS,
#     make_cls_loaders_for_trial,
#     objective_cnn2_classifier,
#     run_study_cnn2,
#     retrain_and_test_cnn2,
#     objective_resnet50csi_classifier,
#     run_study_resnet50csi,
#     retrain_and_test_resnet50csi,
#     train_clip_once,
#     suggest_common_hparams,
#     build_cnn_encoder,
#     build_resnet_encoder,
#     objective_clip_cnn_mac,
#     objective_clip_cnn_mac_loc,
#     objective_clip_resnet_mac,
#     objective_clip_resnet_mac_loc,
#     run_study,
#     retrain_and_test,
# )
from utilities.optuna_helper_functions import (
    # persistence + runner
    save_study_best, run_study,

    # CNN_2
    make_objective_cnn2_classifier, retrain_and_test_cnn2,

    # CLIP objectives (factories) + final eval
    make_objective_clip_cnn_mac,
    make_objective_clip_cnn_mac_loc,
    make_objective_clip_resnet_mac,
    make_objective_clip_resnet_mac_loc,
    retrain_and_test,

    # ResNet50CSI classifier
    make_objective_resnet50csi_classifier,
    retrain_and_test_resnet50csi,
)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################################################################

#%%


# === NEW: CLIP + classifier head ===
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CSI_CLIP_WithClassifier(nn.Module):
    """
    Reuses your CLIP components and adds a linear classifier on the CSI side.
    - zc: CSI embedding (unit norm) -> cls head (num_classes)
    - Contrastive logits: scale * zc @ zt.T  (same as your CSI_CLIP)
    """
    def __init__(self, csi_encoder: nn.Module, label_encoder: nn.Module, num_classes: int):
        super().__init__()
        self.csi = csi_encoder
        self.txt = label_encoder
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))  # same init as yours
        # classification head; we assume encoders output D=proj_dim (e.g., 256)
        # Classify directly from normalized zc (works fine in practice)
        # If you prefer pre-norm features, expose them from encoder and use those instead.
        if isinstance(self.csi, nn.Sequential):
            raise ValueError("csi_encoder shouldn't be a plain Sequential; must return embeddings.")
        # A lazy linear can adapt to whatever proj_dim the encoder returns
        self.cls = nn.LazyLinear(num_classes, bias=True)

    def forward(self, csi_batch, y_batch):
        """
        Returns:
          clip_logits: [B, B] — CLIP similarity matrix for InfoNCE
          cls_logits : [B, C] — supervised logits for CSI->class
          zc, zt     : [B, D] — normalized embeddings (useful for logging/inspection)
        """
        zc = self.csi(csi_batch)        # [B, D], L2-normalized in your encoders
        zt = self.txt(y_batch)          # [B, D], L2-normalized
        scale = self.logit_scale.exp().clamp(max=100.0)
        clip_logits = scale * (zc @ zt.t())   # [B, B]
        cls_logits  = self.cls(zc)            # [B, C]
        return clip_logits, cls_logits, zc, zt

    @torch.no_grad()
    def encode_csi(self, csi_batch):
        return self.csi(csi_batch)

    @torch.no_grad()
    def class_table(self):
        return self.txt.table()
    
    
###############################################################################

# === NEW: multi-task training loop ===
from torch.utils.data import DataLoader

def train_clip_mtl(model,
                   train_ds,
                   val_ds,
                   *,
                   num_classes: int,
                   epochs: int = 10,
                   batch_size: int = 64,
                   lr: float = 1e-3,
                   wd: float = 1e-4,
                   w_clip: float = 1.0,
                   w_cls: float  = 1.0,
                   num_workers: int = 0,
                   device: str = "cuda",
                   print_every: int = 1,
                   evaluate_zero_shot_fn=None):
    """
    Trains CLIP contrastive + supervised classifier together.

    Loss = w_clip * clip_loss(clip_logits) + w_cls * CrossEntropy(cls_logits, y)

    evaluate_zero_shot_fn: function(model, val_loader, device) -> float in [0,1]
      You already have: evaluate_zero_shot(model, val_loader, device)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=num_workers, pin_memory=True)

    opt = torch.optim.AdamW([
        {"params": model.csi.parameters()},
        {"params": model.txt.parameters()},
        {"params": [model.logit_scale]},
        {"params": model.cls.parameters()},
    ], lr=lr, weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()

    best_val_clip = 0.0
    best_state = None

    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            clip_logits, cls_logits, _, _ = model(xb, yb)

            # Your symmetric CLIP loss (mac↔csi both directions)
            loss_clip = clip_loss(clip_logits)
            loss_cls  = ce(cls_logits, yb)

            loss = w_clip * loss_clip + w_cls * loss_cls

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        scheduler.step()
        avg_loss = total / max(1, n_seen)

        # Validation: (1) zero-shot CLIP acc, (2) classifier acc
        model.eval()
        # (1) zero-shot (uses your existing routine)
        val_clip_acc = None
        if evaluate_zero_shot_fn is not None:
            val_clip_acc = evaluate_zero_shot_fn(model, val_loader, device=device)

        # (2) classifier accuracy
        correct = 0
        total   = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                _, cls_logits, _, _ = model(xb, yb)
                pred = cls_logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total   += yb.numel()
        val_cls_acc = correct / max(1, total)

        if (ep % print_every) == 0:
            if val_clip_acc is not None:
                print(f"[Ep {ep:02d}] loss={avg_loss:.4f}  val-CLIP@1={val_clip_acc*100:.2f}%  "
                      f"val-CLS@1={val_cls_acc*100:.2f}%  logit_scale={model.logit_scale.exp().item():.3f}")
            else:
                print(f"[Ep {ep:02d}] loss={avg_loss:.4f}  val-CLS@1={val_cls_acc*100:.2f}%  "
                      f"logit_scale={model.logit_scale.exp().item():.3f}")

        # Track best by zero-shot if available; otherwise by classifier
        metric = val_clip_acc if (val_clip_acc is not None) else val_cls_acc
        if metric is not None and metric > best_val_clip:
            best_val_clip = metric
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_clip

###############################################################################

#%% Training and testing the CSI_CLIP based model, simpleCNN, ResNet50CSI on the same dataset.
# Uses explicit TRAIN / VAL / TEST CSVs (no split). All models see the same samples.

if True:
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = Path(os.getenv("RFF_CSI_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
    SEED = 20250910
    set_global_seed(SEED)
    
    # --------------------------
    # File paths
    # --------------------------
    train_file_path = DATA_DIR / "train_balanced" / "train_balanced_all.csv"
    val_file_path   = DATA_DIR / "val_balanced"   / "val_balanced_all.csv"
    test_file_path  = DATA_DIR / "test_balanced"  / "test_balanced_all.csv"
    
    MAC_ID_LIST = [
        "00:FC:BA:38:4B:00",
        "00:FC:BA:38:4B:01",
        "00:FC:BA:38:4B:02",
        "6C:B2:AE:39:1A:A0",
        "6C:B2:AE:39:1A:A1",
        "70:0F:6A:DE:EC:A0",
        "70:0F:6A:DE:EC:A1",
        "70:0F:6A:DE:EC:A2",
    ]
    
    
    # Cap (per MAC) used by the CSV loader; keep consistent across splits for fairness
    MAX_SAMPLES_PER_MAC_TRAIN = 120_274
    MAX_SAMPLES_PER_MAC_VAL   = 25_023
    MAX_SAMPLES_PER_MAC_TEST  = 26_398
    
    # --------------------------
    # TRAIN set (entire CSV)
    # --------------------------
    # NOTE: set_global_seed above already seeds Python/torch. The CSV loader uses random.sample;
    # this makes the uniform sub-sampling per-MAC reproducible.
    train_data, train_labels = process_csv_fixed_id_uniform_sampling_rssi(
        file_path=train_file_path,
        mac_id_list=MAC_ID_LIST,
        max_samples_per_mac=MAX_SAMPLES_PER_MAC_TRAIN
    )
    assert train_data is not None, "No valid TRAIN data found."
    train_labels = train_labels.long()
    
    # CNN view (TRAIN): [N,1,64,2] + mean_norm()
    x_cnn_train = mean_norm(train_data.unsqueeze(1).float())
    cnn_train_ds = TensorDataset(x_cnn_train, train_labels)
    cnn_train_loader = DataLoader(cnn_train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    
    # CLIP view (TRAIN): [N,2,64] (norm/aug in CSIDataset)
    x_clip_train = train_data.permute(0, 2, 1).float()
    clip_train_ds = CSIDataset(x_clip_train, train_labels, normalize=True, augment=True)
    clip_train_ds.train_mode = True  # enable aug in loader built inside train_clip
    
    print(f"[TRAIN CSV] TRAIN={len(train_labels)}")
    
    # --------------------------
    # VAL set (entire CSV)
    # --------------------------
    # Ensure deterministic uniform sub-sampling per MAC for VAL as well
    random.seed(SEED)
    val_data, val_labels = process_csv_fixed_id_uniform_sampling_rssi(
        file_path=val_file_path,
        mac_id_list=MAC_ID_LIST,
        max_samples_per_mac=MAX_SAMPLES_PER_MAC_VAL
    )
    assert val_data is not None, "No valid VAL data found."
    val_labels = val_labels.long()
    
    # CNN view (VAL)
    x_cnn_val = mean_norm(val_data.unsqueeze(1).float())
    cnn_val_ds = TensorDataset(x_cnn_val, val_labels)
    cnn_val_loader = DataLoader(cnn_val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    
    # CLIP view (VAL)
    x_clip_val = val_data.permute(0, 2, 1).float()
    clip_val_ds = CSIDataset(x_clip_val, val_labels, normalize=True, augment=False)
    clip_val_ds.train_mode = False
    
    print(f"[VAL CSV]   VAL={len(val_labels)}")
    
    # --------------------------
    # TEST set (entire CSV) — shared for all models
    # --------------------------
    random.seed(SEED)
    test_data, test_labels = process_csv_fixed_id_uniform_sampling_rssi(
        file_path=test_file_path,
        mac_id_list=MAC_ID_LIST,
        max_samples_per_mac=MAX_SAMPLES_PER_MAC_TEST
    )
    assert test_data is not None and test_labels is not None, "No valid TEST data found."
    test_labels = test_labels.long()
    
    # CNN view (TEST)
    x_cnn_test = mean_norm(test_data.unsqueeze(1).float())
    cnn_test_ds = TensorDataset(x_cnn_test, test_labels)
    cnn_test_loader = DataLoader(cnn_test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    
    # CLIP view (TEST)
    x_clip_test = test_data.permute(0, 2, 1).float()
    clip_test_ds = CSIDataset(x_clip_test, test_labels, normalize=True, augment=False)
    clip_test_ds.train_mode = False
    clip_test_loader = DataLoader(clip_test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"[TEST CSV]  TEST={len(test_labels)} (shared for all models)")
    
    
##############################################################################

#%% CSI_CLIP with CNN_2 CSI enoder with MAC hex -> learnable projector 

if False:
    # ======= Reuse EXACT SAME loaders/splits you already created =======
    # clip_train_ds, clip_val_ds, clip_test_loader come from your code.
    
    # Provide MAC → (x,y,z) dict (meters), keys must match MAC_ID_LIST entries:
    mac_to_xyz = {
        "00:FC:BA:38:4B:00": (19.61, 18.60, 4),
        "00:FC:BA:38:4B:01": (19.61, 18.60, 4),
        "00:FC:BA:38:4B:02": (19.61, 18.60, 4),
        "6C:B2:AE:39:1A:A0": (17.83, 11.30, 4),
        "6C:B2:AE:39:1A:A1": (17.83, 11.30, 4),
        "70:0F:6A:DE:EC:A0": (15.66, 11.30, 4),
        "70:0F:6A:DE:EC:A1": (15.66, 11.30, 4),
        "70:0F:6A:DE:EC:A2": (15.66, 11.30, 4),
    }
    
    num_classes = len(MAC_ID_LIST)
    
    # ---------------------------
    # (1) CSI = 2-layer CNN; Label = MAC hex → learnable projector
    # ---------------------------
    model_1 = CSI_CLIP(
        csi_encoder=CSIEncoder(in_ch=2, proj_dim=256),                 # REUSED encoder (CNN) :contentReference[oaicite:7]{index=7}
        label_encoder=LabelHexProjector(MAC_ID_LIST, dim=256, hex_dim=64)  # NEW
    )
    trained_1 = train_clip(model_1, clip_train_ds, clip_val_ds,
                           epochs=10, batch_size=64, lr=1e-3, wd=1e-4,
                           num_workers=0, device=("cuda" if torch.cuda.is_available() else "cpu"))  # REUSED loop :contentReference[oaicite:8]{index=8}
    acc_1 = evaluate_zero_shot(trained_1, clip_test_loader, device=("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"(1) CNN + MAC(hex→learnable) TEST top-1: {acc_1*100:.2f}%")

#%%

# --- Example: CNN CSI encoder + MAC hex label encoder, multi-task ---
num_classes = len(MAC_ID_LIST)

csi_side   = CSIEncoder(in_ch=2, proj_dim=256)                     # REUSED
label_side = LabelHexProjector(MAC_ID_LIST, dim=256, hex_dim=64)   # REUSED

model_mtl = CSI_CLIP_WithClassifier(
    csi_encoder=csi_side,
    label_encoder=label_side,
    num_classes=num_classes
)

# Train both losses together
model_mtl, _ = train_clip_mtl(
    model=model_mtl,
    train_ds=clip_train_ds,
    val_ds=clip_val_ds,
    num_classes=num_classes,
    epochs=20,
    batch_size=64,
    lr=1e-3,
    wd=1e-4,
    w_clip=1.0,      # weight on CLIP contrastive
    w_cls=1.0,       # weight on classifier CE
    device=("cuda" if torch.cuda.is_available() else "cpu"),
    evaluate_zero_shot_fn=evaluate_zero_shot  # REUSED
)

# Evaluate both heads on the TEST set
from torch.utils.data import DataLoader
test_loader = DataLoader(clip_test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# (A) zero-shot accuracy (CLIP retrieval)
zs_acc = evaluate_zero_shot(model_mtl, test_loader, device=device)
print(f"[MTL] TEST zero-shot top-1: {zs_acc*100:.2f}%")

# (B) classifier accuracy (direct CSI→class)
model_mtl.eval()
correct = 0; total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device); yb = yb.to(device)
        _, cls_logits, _, _ = model_mtl(xb, yb)
        pred = cls_logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
cls_acc = correct / max(1, total)
print(f"[MTL] TEST classifier top-1: {cls_acc*100:.2f}%")
