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

_PROJECT_ROOT = Path(__file__).resolve().parents[1] # folder that contains /utilities and /models
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class CSIEncoder3(nn.Module):
    """
    3-layer 1D CNN encoder for CSI shaped [B, 2, 64].
    - backbone: Conv1d(2->64)->ReLU -> Conv1d(64->128)->ReLU -> Conv1d(128->192)->ReLU
    - forward_features(): returns [B, 192, L] (L≈64 with same padding)
    - forward(): GAP -> Linear(192->proj_dim) -> LayerNorm -> L2 normalize  (CLIP embedding)
    """
    def __init__(self, in_ch: int = 2, proj_dim: int = 256, use_bn: bool = False, dropout: float = 0.0):
        super().__init__()
        Conv = nn.Conv1d
        layers = []
        def block(c_in, c_out):
            m = [Conv(c_in, c_out, kernel_size=5, padding=2), nn.ReLU(inplace=True)]
            if use_bn: m.insert(1, nn.BatchNorm1d(c_out))
            return m
        layers += block(in_ch, 64)
        layers += block(64, 128)
        layers += block(128, 192)
        self.backbone = nn.Sequential(*layers)
        self.pool     = nn.AdaptiveAvgPool1d(1)
        self.dropout  = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.proj     = nn.Linear(192, proj_dim)
        self.norm     = nn.LayerNorm(proj_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, 64]  ->  [B, 192, 64]
        return self.backbone(x.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fm = self.forward_features(x)          # [B, 192, L]
        h  = self.pool(fm).squeeze(-1)         # [B, 192]
        h  = self.dropout(h)
        z  = self.norm(self.proj(h))           # [B, D]
        return F.normalize(z, dim=-1)

import math

class CSI_CLIP_WithConvClassifier3(nn.Module):
    """
    CLIP-style model (contrastive CSI↔label) + 3-layer conv-only classifier head.
    - Uses CSIEncoder3.forward_features() for the classifier path.
    - contrastive path identical to CSI_CLIP.
    """
    def __init__(self, csi_encoder: nn.Module, label_encoder: nn.Module, num_classes: int,
                 head_hidden=(256, 256, 256), head_dropout: float = 0.0, head_bn: bool = False):
        super().__init__()
        self.csi = csi_encoder          # expect CSIEncoder3 (or compatible with forward_features)
        self.txt = label_encoder
        self.cls_head = ConvOnlyClassifier1D_3L(num_classes, hidden=head_hidden,
                                                dropout=head_dropout, use_bn=head_bn)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))

    def forward(self, x_bcl: torch.Tensor, y_idx: torch.Tensor):
        feat_map   = self.csi.forward_features(x_bcl)   # [B, 192, L]
        cls_logits = self.cls_head(feat_map)            # [B, C]

        zc = self.csi(x_bcl)                            # [B, D] (proj_dim, L2 norm)
        zt = self.txt(y_idx)                            # [B, D] (L2 norm)
        scale = self.logit_scale.exp().clamp(max=100.0)
        clip_logits = scale * (zc @ zt.t())             # [B, B]

        return clip_logits, cls_logits, zc, zt



class ConvOnlyClassifier1D_3L(nn.Module):
    """
    3-layer conv head over encoder feature map [B, 192, L]:
      Conv1d(192->256, k=3) -> ReLU
      Conv1d(256->256, k=3) -> ReLU
      Conv1d(256->256, k=3) -> ReLU
      GAP -> Conv1d(256->C, k=1) -> squeeze -> [B, C]
    """
    def __init__(self, num_classes: int, hidden=(256, 256, 256), dropout: float = 0.0, use_bn: bool = False):
        super().__init__()
        c1, c2, c3 = hidden
        def cb(c_in, c_out):
            m = [nn.Conv1d(c_in, c_out, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            if use_bn: m.insert(1, nn.BatchNorm1d(c_out))
            return m
        self.conv = nn.Sequential(
            *cb(192, c1),
            *cb(c1,  c2),
            *cb(c2,  c3),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(c3, num_classes, kernel_size=1)
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        return self.conv(feat_map).squeeze(-1)   # [B, C]



# --- PATCHED: CSIEncoder with forward_features() ---
class CSIEncoder(nn.Module):
    """
    1D Conv encoder for CSI shaped [B, 2, 64].
    Now exposes forward_features() -> [B, 128, L] so a conv head can classify
    without a Linear layer.
    """
    def __init__(self, in_ch=2, proj_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=64,  kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(in_channels=64,    out_channels=128, kernel_size=5, padding=2), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)     # -> [B, 128, 1]
        self.proj = nn.Linear(128, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the pre-pooled feature map: [B, 128, L] (L ≈ 64 here).
        """
        return self.backbone(x.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fm = self.forward_features(x)           # [B, 128, L]
        h  = self.pool(fm).squeeze(-1)          # [B, 128]
        z  = self.norm(self.proj(h))            # [B, D]
        return F.normalize(z, dim=-1)           # unit norm

class ConvOnlyClassifier1D(nn.Module):
    """
    All-conv classifier:
      [B,128,L] -> Conv1d(128,256,3) -> ReLU
                 -> Conv1d(256,256,3) -> ReLU
                 -> GAP -> [B,256,1]
                 -> Conv1d(256,num_classes,1) -> [B,C,1] -> squeeze -> [B,C]
    """
    def __init__(self, num_classes: int, hidden=(256, 256), dropout: float = 0.0):
        super().__init__()
        h1, h2 = hidden
        self.conv = nn.Sequential(
            nn.Conv1d(128, h1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(h1,  h2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),                 # -> [B, h2, 1]
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(h2, num_classes, kernel_size=1)  # 1x1 conv replaces Linear
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        # feat_map: [B, 128, L]
        logits = self.conv(feat_map).squeeze(-1)  # [B, C]
        return logits

import math

class CSI_CLIP_WithConvClassifier(nn.Module):
    """
    CLIP contrastive branch + conv-only classifier head (no Linear).
    - Uses CSIEncoder.forward_features() for the conv head.
    - Contrastive head identical to your CSI_CLIP.
    """
    def __init__(self, csi_encoder: nn.Module, label_encoder: nn.Module, num_classes: int,
                 head_hidden=(256, 256), head_dropout: float = 0.0):
        super().__init__()
        self.csi = csi_encoder
        self.txt = label_encoder
        self.cls_head = ConvOnlyClassifier1D(num_classes, hidden=head_hidden, dropout=head_dropout)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))

    def forward(self, x_bcl: torch.Tensor, y_idx: torch.Tensor):
        # Conv-only classifier path
        feat_map   = self.csi.forward_features(x_bcl)  # [B,128,L]
        cls_logits = self.cls_head(feat_map)           # [B,C]

        # CLIP path
        zc = self.csi(x_bcl)                           # [B,D] (unit norm)
        zt = self.txt(y_idx)                           # [B,D] (unit norm)
        scale = self.logit_scale.exp().clamp(max=100.0)
        clip_logits = scale * (zc @ zt.t())            # [B,B]

        return clip_logits, cls_logits, zc, zt




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

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

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
    Multi-task trainer for CLIP(+optional classifier) models.

    Loss = w_clip * clip_loss(clip_logits) + w_cls * CrossEntropy(cls_logits, y)

    Model forward flexibility:
      - If model returns (clip_logits, cls_logits, ...), both heads are used.
      - If returns a single tensor:
          * if [B,B] (square) => treat as clip_logits
          * else              => treat as classifier logits
      - If returns only cls_logits, set w_clip=0.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=num_workers, pin_memory=True)

    # ---- Param groups (robust) ----
    param_groups = []
    if hasattr(model, "csi"):          param_groups.append({"params": model.csi.parameters()})
    if hasattr(model, "txt"):          param_groups.append({"params": model.txt.parameters()})
    if hasattr(model, "logit_scale"):  param_groups.append({"params": [model.logit_scale]})
    if hasattr(model, "cls"):          param_groups.append({"params": model.cls.parameters()})
    if hasattr(model, "cls_head"):     param_groups.append({"params": model.cls_head.parameters()})
    param_groups = [g for g in param_groups if any(p.requires_grad for p in g["params"])]
    assert len(param_groups) > 0, "No trainable parameters found!"

    opt = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    from utilities.losses import clip_loss

    best_metric = -1.0
    best_state  = None
    best_metrics_snapshot = {"val_clip_acc": None, "val_cls_acc": None}

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            out = model(xb, yb)

            clip_logits, cls_logits = None, None
            if isinstance(out, (list, tuple)):
                if len(out) >= 2:
                    clip_logits, cls_logits = out[0], out[1]
                elif len(out) == 1:
                    # single tensor from a tuple – assume classifier logits
                    cls_logits = out[0]
            else:
                # single tensor: disambiguate by shape
                if out.dim() == 2 and out.shape[0] == out.shape[1] == xb.shape[0]:
                    clip_logits = out          # [B,B]
                else:
                    cls_logits  = out          # [B,C] (or similar)

            loss = 0.0
            if (w_clip > 0.0) and (clip_logits is not None):
                loss = loss + w_clip * clip_loss(clip_logits)
            if (w_cls  > 0.0) and (cls_logits  is not None):
                loss = loss + w_cls  * ce(cls_logits, yb)

            if loss == 0.0:
                raise RuntimeError(
                    "No loss terms active. Ensure w_clip>0 with clip logits available and/or w_cls>0 with cls logits."
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)
            total_seen += xb.size(0)

        scheduler.step()
        avg_loss = total_loss / max(1, total_seen)

        # ---- Validation ----
        model.eval()
        # (A) zero-shot CLIP acc (optional)
        val_clip_acc = None
        can_eval_clip = (evaluate_zero_shot_fn is not None) and hasattr(model, "txt") and hasattr(model, "csi")
        if can_eval_clip:
            try:
                val_clip_acc = evaluate_zero_shot_fn(model, val_loader, device=device)
            except Exception:
                val_clip_acc = None

        # (B) classifier acc (if head/logits exist)
        val_cls_acc = None
        want_cls_eval = hasattr(model, "cls") or hasattr(model, "cls_head") or (w_cls > 0.0)
        if want_cls_eval:
            correct = 0
            total   = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    out = model(xb, yb)

                    cls_logits = None
                    if isinstance(out, (list, tuple)):
                        if len(out) >= 2:
                            cls_logits = out[1]
                        elif len(out) == 1:
                            cls_logits = out[0]
                    else:
                        # single tensor: if it's square [B,B], it's clip-only → skip
                        if not (out.dim() == 2 and out.shape[0] == out.shape[1] == xb.shape[0]):
                            cls_logits = out

                    if cls_logits is None:
                        continue
                    pred = cls_logits.argmax(dim=1)
                    correct += (pred == yb).sum().item()
                    total   += yb.numel()
            if total > 0:
                val_cls_acc = correct / total

        if (ep % print_every) == 0:
            parts = [f"[Ep {ep:02d}] loss={avg_loss:.4f}"]
            if val_clip_acc is not None:
                parts.append(f"val-CLIP@1={val_clip_acc*100:.2f}%")
            if val_cls_acc is not None:
                parts.append(f"val-CLS@1={val_cls_acc*100:.2f}%")
            if hasattr(model, "logit_scale"):
                parts.append(f"logit_scale={model.logit_scale.exp().item():.3f}")
            print("  ".join(parts))

        # selection metric: prefer zero-shot if available; else classifier; else inverse loss
        if val_clip_acc is not None:
            metric_now = val_clip_acc
        elif val_cls_acc is not None:
            metric_now = val_cls_acc
        else:
            metric_now = 1.0 / (1e-9 + avg_loss)

        if metric_now > best_metric:
            best_metric = metric_now
            best_state  = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics_snapshot = {"val_clip_acc": val_clip_acc, "val_cls_acc": val_cls_acc}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_metrics_snapshot


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

    #%% mac_to_xyz

if True:
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


#%%
# ---- build CLIP (no classifier needed here) ----
# Build the parts
num_classes = len(MAC_ID_LIST)
csi_side   = CSIEncoder3(in_ch=2, proj_dim=256, use_bn=False, dropout=0.0)  # NEW 3-layer encoder
label_side = LabelHexProjector(MAC_ID_LIST, dim=256, hex_dim=64)            # or LabelHexPlusLoc(...)

model = CSI_CLIP_WithConvClassifier3(
    csi_encoder=csi_side,
    label_encoder=label_side,
    num_classes=num_classes,
    head_hidden=(256,256,256), head_dropout=0.0, head_bn=False
)

# Train both losses together
model, best = train_clip_mtl(
    model=model,
    train_ds=clip_train_ds,
    val_ds=clip_val_ds,
    num_classes=num_classes,
    epochs=10, batch_size=64, lr=1e-3, wd=1e-4,
    w_clip=1.0, w_cls=1.0,                           # multitask; or set w_clip=0.0 for classifier-only
    device=("cuda" if torch.cuda.is_available() else "cpu"),
    evaluate_zero_shot_fn=evaluate_zero_shot         # or None if classifier-only
)

# Test classifier head
from torch.utils.data import DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"
test_loader = DataLoader(clip_test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

model.eval(); correct=total=0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        _, cls_logits, _, _ = model(xb, yb)
        pred = cls_logits.argmax(1)
        correct += (pred==yb).sum().item(); total += yb.numel()
print(f"[3-layer enc + 3-layer head] TEST top-1: {100*correct/max(1,total):.2f}%")

