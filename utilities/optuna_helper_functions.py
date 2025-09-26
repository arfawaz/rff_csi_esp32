# # -*- coding: utf-8 -*-
# """
# Created on Wed Sep 24 17:09:08 2025

# @author: fawaz243
# """
import optuna
from optuna.pruners import MedianPruner
from torch.utils.data import DataLoader

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


import json, datetime
from pathlib import Path
import optuna
from optuna.pruners import MedianPruner
from torch.utils.data import DataLoader
SEED = 20250910


# ====================== JSON + persistence (no globals) ======================

def _jsonify(o):
    import json as _json
    try:
        _json.dumps(o)
        return o
    except Exception:
        if isinstance(o, dict):
            return {k: _jsonify(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_jsonify(v) for v in o]
        if hasattr(o, "item"):  # numpy/torch scalar
            return o.item()
        return str(o)

def save_study_best(
    study,
    study_name: str,
    outdir: Path,
    *,
    seed: int,
    extra: dict | None = None,
    save_trials_csv: bool = True,
):
    """Save best params/value to JSON; optionally dump trials to CSV."""
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "study_name": study_name,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "best_value_val_acc": float(study.best_value),  # in [0,1]
        "best_params": _jsonify(study.best_trial.params),
        "seed": _jsonify(seed),
        **(_jsonify(extra) if extra else {}),
    }
    json_path = outdir / f"{study_name}_best.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[SAVE] {json_path}")

    if save_trials_csv:
        try:
            df = study.trials_dataframe(attrs=(
                "number","value","params","state","datetime_start","datetime_complete"
            ))
            csv_path = outdir / f"{study_name}_trials.csv"
            df.to_csv(csv_path, index=False)
            print(f"[SAVE] {csv_path}")
        except Exception as e:
            print(f"[WARN] Could not save trials CSV for {study_name}: {e}")

def save_checkpoint(
    model,
    study_name: str,
    outdir: Path,
    *,
    save_checkpoints: bool = True,
):
    if not save_checkpoints:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = outdir / f"{study_name}_best.ckpt"
    torch.save({"model": model.state_dict()}, ckpt_path)
    print(f"[SAVE] {ckpt_path}")


# ============================ Loader utilities ================================
def make_cls_loaders_for_trial(
    trial,
    *,
    has_datasets: bool,
    train_ds=None,
    val_ds=None,
    test_ds=None,
    fixed_train_loader=None,
    fixed_val_loader=None,
    fixed_test_loader=None,
):
    """
    If has_datasets=True: rebuild loaders with tunable batch size (suggested by trial).
    Else: reuse given fixed_*_loader and return them; batch size = None.
    """
    if not has_datasets:
        assert fixed_train_loader and fixed_val_loader and fixed_test_loader, \
            "fixed_*_loader must be provided when has_datasets=False"
        return fixed_train_loader, fixed_val_loader, fixed_test_loader, None

    bs = trial.suggest_categorical("batch_size", [32, 64, 128])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0,
                              pin_memory=True, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0,
                              pin_memory=True, drop_last=False)
    return train_loader, val_loader, test_loader, bs


# ====================== CNN_2 classifier: objective factory ===================
def make_objective_cnn2_classifier(
    *,
    mac_id_list: list[str],
    device,
    has_datasets: bool,
    train_ds=None, val_ds=None, test_ds=None,
    fixed_train_loader=None, fixed_val_loader=None, fixed_test_loader=None,
):
    """
    Returns an Optuna objective(trial) that trains/evals CNN_2.
    Avoids globals by closing over datasets/loaders/device/mac list.
    """
    num_classes = len(mac_id_list)

    def objective(trial):
        proj_dim = trial.suggest_categorical("proj_dim", [128, 256, 384])
        l2norm   = trial.suggest_categorical("l2norm",  [True, False])
        epochs   = trial.suggest_int("epochs", 5, 50)
        lr       = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        wd       = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        opt_name = trial.suggest_categorical("optimizer", ["adam", "adamw"])

        tl, vl, _testl, _bs = make_cls_loaders_for_trial(
            trial,
            has_datasets=has_datasets,
            train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
            fixed_train_loader=fixed_train_loader,
            fixed_val_loader=fixed_val_loader,
            fixed_test_loader=fixed_test_loader,
        )

        model = CNN_2(num_classes=num_classes, proj_dim=proj_dim, l2norm=l2norm).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = (optim.Adam if opt_name == "adam" else optim.AdamW)(
            model.parameters(), lr=lr, weight_decay=wd
        )

        train(model=model, train_loader=tl, test_loader=vl,
              criterion=criterion, optimizer=optimizer, num_epochs=epochs)

        val_pct = test(model, vl)  # returns percent
        return val_pct / 100.0

    return objective


def retrain_and_test_cnn2(
    best_params: dict,
    *,
    mac_id_list: list[str],
    device,
    has_datasets: bool,
    outdir: Path,
    save_checkpoints: bool = True,
    # datasets if has_datasets=True:
    train_ds=None, val_ds=None, test_ds=None,
    # fixed loaders if has_datasets=False:
    fixed_train_loader=None, fixed_val_loader=None, fixed_test_loader=None,
):
    """Final fit CNN_2 with best params, evaluate on TEST; save checkpoint."""
    num_classes = len(mac_id_list)

    if has_datasets:
        bs = best_params.get("batch_size", 64)
        tl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0, pin_memory=True, drop_last=True)
        vl = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
        te = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    else:
        tl, vl, te = fixed_train_loader, fixed_val_loader, fixed_test_loader

    proj_dim = best_params.get("proj_dim", 256)
    l2norm   = best_params.get("l2norm", True)
    epochs   = max(best_params.get("epochs", 10), 10)
    lr       = best_params.get("lr", 1e-3)
    wd       = best_params.get("weight_decay", 1e-4)
    opt_name = best_params.get("optimizer", "adamw")

    model = CNN_2(num_classes=num_classes, proj_dim=proj_dim, l2norm=l2norm).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = (optim.Adam if opt_name == "adam" else optim.AdamW)(
        model.parameters(), lr=lr, weight_decay=wd
    )

    train(model=model, train_loader=tl, test_loader=vl,
          criterion=criterion, optimizer=optimizer, num_epochs=epochs)

    test_pct = test(model, te)
    print(f"[CNN_2 BEST] TEST top-1: {test_pct:.2f}%")
    save_checkpoint(model, "CNN2_CLASSIFIER", outdir, save_checkpoints=save_checkpoints)
    return test_pct / 100.0


# ============ ResNet50CSI classifier: objective factory & retrain =============
def make_objective_resnet50csi_classifier(
    *,
    mac_id_list: list[str],
    device,
    has_datasets: bool,
    train_ds=None, val_ds=None,
    fixed_train_loader=None, fixed_val_loader=None,
):
    num_classes = len(mac_id_list)

    def objective(trial):
        epochs    = trial.suggest_int("epochs", 1, 12)
        lr        = trial.suggest_float("lr", 1e-6, 3e-4, log=True)
        wd        = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        opt_name  = trial.suggest_categorical("optimizer", ["adam", "adamw"])
        pretrained = trial.suggest_categorical("pretrained", [True, False])

        if has_datasets:
            bs = trial.suggest_categorical("batch_size", [32, 64, 128])
            tl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0, pin_memory=True, drop_last=True)
            vl = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
        else:
            tl, vl = fixed_train_loader, fixed_val_loader

        model = ResNet50CSI(num_classes=num_classes, pretrained=pretrained).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = (optim.Adam if opt_name == "adam" else optim.AdamW)(
            model.parameters(), lr=lr, weight_decay=wd
        )

        train(model=model, train_loader=tl, test_loader=vl,
              criterion=criterion, optimizer=optimizer, num_epochs=epochs)
        val_pct = test(model, vl)
        return val_pct / 100.0

    return objective


def retrain_and_test_resnet50csi(
    best_params: dict,
    *,
    mac_id_list: list[str],
    device,
    has_datasets: bool,
    outdir: Path,
    save_checkpoints: bool = True,
    # datasets if has_datasets=True:
    train_ds=None, val_ds=None, test_ds=None,
    # fixed loaders if has_datasets=False:
    fixed_train_loader=None, fixed_val_loader=None, fixed_test_loader=None,
):
    num_classes = len(mac_id_list)
    if has_datasets:
        bs = best_params.get("batch_size", 64)
        tl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0, pin_memory=True, drop_last=True)
        vl = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
        te = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    else:
        tl, vl, te = fixed_train_loader, fixed_val_loader, fixed_test_loader

    epochs     = max(best_params.get("epochs", 6), 6)
    lr         = best_params.get("lr", 1e-4)
    wd         = best_params.get("weight_decay", 1e-4)
    opt_name   = best_params.get("optimizer", "adamw")
    pretrained = best_params.get("pretrained", True)

    model = ResNet50CSI(num_classes=num_classes, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = (optim.Adam if opt_name == "adam" else optim.AdamW)(
        model.parameters(), lr=lr, weight_decay=wd
    )

    train(model=model, train_loader=tl, test_loader=vl,
          criterion=criterion, optimizer=optimizer, num_epochs=epochs)

    test_pct = test(model, te)
    print(f"[ResNet50CSI BEST] TEST top-1: {test_pct:.2f}%")
    save_checkpoint(model, "RESNET50CSI_CLASSIFIER", outdir, save_checkpoints=save_checkpoints)
    return test_pct / 100.0


# =============================== CLIP helpers =================================
def train_clip_once(model, *, clip_train_ds, clip_val_ds, epochs, batch_size, lr, wd, device):
    trained = train_clip(model, clip_train_ds, clip_val_ds,
                         epochs=epochs, batch_size=batch_size, lr=lr, wd=wd,
                         num_workers=0, device=device)
    val_loader = DataLoader(clip_val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_acc = evaluate_zero_shot(trained, val_loader, device=device)
    return trained, val_acc

def suggest_common_hparams(trial, *, is_resnet: bool):
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs     = trial.suggest_int("epochs", 4, 50)
    lr         = trial.suggest_float("lr", 1e-5 if is_resnet else 5e-5, 5e-4 if is_resnet else 3e-3, log=True)
    wd         = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    proj_dim   = trial.suggest_categorical("proj_dim", [128, 256, 384])
    return batch_size, epochs, lr, wd, proj_dim

def build_cnn_encoder(proj_dim):
    return CSIEncoder(in_ch=2, proj_dim=proj_dim)

def build_resnet_encoder(proj_dim, trial):
    freeze_bn = trial.suggest_categorical("freeze_backbone_bn", [False, True])
    return CSIResNet50Encoder(proj_dim=proj_dim, pretrained=True, freeze_backbone_bn=freeze_bn)


# ======================= CLIP objectives: factories ===========================
def make_objective_clip_cnn_mac(
    *,
    mac_id_list: list[str],
    device,
    clip_train_ds,
    clip_val_ds,
):
    def objective(trial):
        batch_size, epochs, lr, wd, proj_dim = suggest_common_hparams(trial, is_resnet=False)
        hex_dim = trial.suggest_categorical("hex_dim", [32, 64, 128])

        label_side = LabelHexProjector(mac_id_list, dim=proj_dim, hex_dim=hex_dim)
        csi_side   = build_cnn_encoder(proj_dim)
        model      = CSI_CLIP(csi_encoder=csi_side, label_encoder=label_side).to(device)

        _, val_acc = train_clip_once(model, clip_train_ds=clip_train_ds, clip_val_ds=clip_val_ds,
                                     epochs=epochs, batch_size=batch_size, lr=lr, wd=wd,
                                     device=("cuda" if torch.cuda.is_available() else "cpu"))
        return val_acc
    return objective

def make_objective_clip_cnn_mac_loc(
    *,
    mac_id_list: list[str],
    mac_to_xyz: dict[str, tuple[float,float,float]],
    seed: int,
    device,
    clip_train_ds,
    clip_val_ds,
):
    def objective(trial):
        batch_size, epochs, lr, wd, proj_dim = suggest_common_hparams(trial, is_resnet=False)
        hex_dim    = trial.suggest_categorical("hex_dim", [32, 64, 128])
        lambda_pos = trial.suggest_float("lambda_pos", 0.1, 3.0, log=True)
        sigma      = trial.suggest_float("rff_sigma", 0.5, 3.0, log=True)
        coord_s    = trial.suggest_float("coord_scale", 0.5, 3.0, log=True)

        label_side = LabelHexPlusLoc(mac_id_list, mac_to_xyz, dim=proj_dim, hex_dim=hex_dim,
                                     lambda_pos=lambda_pos, sigma=sigma, seed=seed, coord_scale=coord_s)
        csi_side   = build_cnn_encoder(proj_dim)
        model      = CSI_CLIP(csi_encoder=csi_side, label_encoder=label_side).to(device)

        _, val_acc = train_clip_once(model, clip_train_ds=clip_train_ds, clip_val_ds=clip_val_ds,
                                     epochs=epochs, batch_size=batch_size, lr=lr, wd=wd,
                                     device=("cuda" if torch.cuda.is_available() else "cpu"))
        return val_acc
    return objective

def make_objective_clip_resnet_mac(
    *,
    mac_id_list: list[str],
    device,
    clip_train_ds,
    clip_val_ds,
):
    def objective(trial):
        batch_size, epochs, lr, wd, proj_dim = suggest_common_hparams(trial, is_resnet=True)
        hex_dim = trial.suggest_categorical("hex_dim", [32, 64, 128])

        label_side = LabelHexProjector(mac_id_list, dim=proj_dim, hex_dim=hex_dim)
        csi_side   = build_resnet_encoder(proj_dim, trial)
        model      = CSI_CLIP(csi_encoder=csi_side, label_encoder=label_side).to(device)

        _, val_acc = train_clip_once(model, clip_train_ds=clip_train_ds, clip_val_ds=clip_val_ds,
                                     epochs=epochs, batch_size=batch_size, lr=lr, wd=wd,
                                     device=("cuda" if torch.cuda.is_available() else "cpu"))
        return val_acc
    return objective

def make_objective_clip_resnet_mac_loc(
    *,
    mac_id_list: list[str],
    mac_to_xyz: dict[str, tuple[float,float,float]],
    seed: int,
    device,
    clip_train_ds,
    clip_val_ds,
):
    def objective(trial):
        batch_size, epochs, lr, wd, proj_dim = suggest_common_hparams(trial, is_resnet=True)
        hex_dim    = trial.suggest_categorical("hex_dim", [32, 64, 128])
        lambda_pos = trial.suggest_float("lambda_pos", 0.1, 3.0, log=True)
        sigma      = trial.suggest_float("rff_sigma", 0.5, 3.0, log=True)
        coord_s    = trial.suggest_float("coord_scale", 0.5, 3.0, log=True)

        label_side = LabelHexPlusLoc(mac_id_list, mac_to_xyz, dim=proj_dim, hex_dim=hex_dim,
                                     lambda_pos=lambda_pos, sigma=sigma, seed=seed, coord_scale=coord_s)
        csi_side   = build_resnet_encoder(proj_dim, trial)
        model      = CSI_CLIP(csi_encoder=csi_side, label_encoder=label_side).to(device)

        _, val_acc = train_clip_once(model, clip_train_ds=clip_train_ds, clip_val_ds=clip_val_ds,
                                     epochs=epochs, batch_size=batch_size, lr=lr, wd=wd,
                                     device=("cuda" if torch.cuda.is_available() else "cpu"))
        return val_acc
    return objective


# =================== shared study runner & final CLIP eval ====================
def run_study(objective_fn, study_name: str, n_trials: int, *, seed: int):
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    pruner  = MedianPruner(n_startup_trials=max(5, min(10, n_trials//3)))
    study   = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=study_name)
    study.optimize(objective_fn, n_trials=n_trials, gc_after_trial=True)
    print(f"[{study_name}] Best val@1 = {study.best_value*100:.2f}%")
    print(f"[{study_name}] Best params = {study.best_trial.params}")
    return study

def retrain_and_test(
    best_params: dict,
    *,
    is_resnet: bool,
    use_location: bool,
    mac_id_list: list[str],
    mac_to_xyz: dict[str, tuple[float,float,float]] | None,
    seed: int,
    device,
    clip_train_ds,
    clip_val_ds,
    clip_test_loader,
    outdir: Path,
    save_checkpoints: bool = True,
):
    batch_size = best_params["batch_size"]
    epochs     = max(best_params["epochs"], 10)
    lr         = best_params["lr"]
    wd         = best_params["weight_decay"]
    proj_dim   = best_params["proj_dim"]

    if use_location:
        hex_dim    = best_params["hex_dim"]
        lambda_pos = best_params["lambda_pos"]
        sigma      = best_params["rff_sigma"]
        coord_s    = best_params["coord_scale"]
        assert mac_to_xyz is not None, "mac_to_xyz must be provided when use_location=True"
        label_side = LabelHexPlusLoc(
            mac_id_list, mac_to_xyz, dim=proj_dim, hex_dim=hex_dim,
            lambda_pos=lambda_pos, sigma=sigma, seed=seed, coord_scale=coord_s
        )
    else:
        label_side = LabelHexProjector(mac_id_list, dim=proj_dim, hex_dim=best_params["hex_dim"])

    if is_resnet:
        freeze_bn = best_params.get("freeze_backbone_bn", False)
        csi_side  = CSIResNet50Encoder(proj_dim=proj_dim, pretrained=True, freeze_backbone_bn=freeze_bn)
    else:
        csi_side  = CSIEncoder(in_ch=2, proj_dim=proj_dim)

    model = CSI_CLIP(csi_encoder=csi_side, label_encoder=label_side).to(device)
    model, _ = train_clip_once(model, clip_train_ds=clip_train_ds, clip_val_ds=clip_val_ds,
                               epochs=epochs, batch_size=batch_size, lr=lr, wd=wd,
                               device=("cuda" if torch.cuda.is_available() else "cpu"))
    test_acc = evaluate_zero_shot(model, clip_test_loader, device=("cuda" if torch.cuda.is_available() else "cpu"))
    save_checkpoint(model, "CLIP_FINAL", outdir, save_checkpoints=save_checkpoints)
    return test_acc

