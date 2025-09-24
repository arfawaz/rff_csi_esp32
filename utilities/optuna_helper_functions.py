# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 17:09:08 2025

@author: fawaz243
"""
import optuna
from optuna.pruners import MedianPruner
from torch.utils.data import DataLoader
SEED = 20250910

def _jsonify(o):
    # make optuna/numpy types JSON friendly
    try:
        json.dumps(o)
        return o
    except Exception:
        if isinstance(o, dict):
            return {k: _jsonify(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_jsonify(v) for v in o]
        if hasattr(o, "item"):  # numpy scalar / torch scalar
            return o.item()
        return str(o)

def save_study_best(study, study_name: str, extra: dict | None = None):
    """Save best params + best value to JSON; optionally dump trials to CSV."""
    payload = {
        "study_name": study_name,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "best_value_val_acc": float(study.best_value),  # in [0,1]
        "best_params": _jsonify(study.best_trial.params),
        "seed": _jsonify(SEED),
        **(_jsonify(extra) if extra else {}),
    }
    json_path = HPO_OUTDIR / f"{study_name}_best.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[SAVE] {json_path}")

    if SAVE_TRIALS_CSV:
        try:
            df = study.trials_dataframe(attrs=("number","value","params","state","datetime_start","datetime_complete"))
            csv_path = HPO_OUTDIR / f"{study_name}_trials.csv"
            df.to_csv(csv_path, index=False)
            print(f"[SAVE] {csv_path}")
        except Exception as e:
            print(f"[WARN] Could not save trials CSV for {study_name}: {e}")

def save_checkpoint(model, study_name: str):
    if not SAVE_CHECKPOINTS:
        return
    ckpt_path = HPO_OUTDIR / f"{study_name}_best.ckpt"
    torch.save({"model": model.state_dict()}, ckpt_path)
    print(f"[SAVE] {ckpt_path}")

# ---- Toggle if you have datasets in scope (to tune batch size for CNN_2) ----
HAS_DATASETS = True  # set to False if only cnn_*_loader are available

# ------------------------------ Helper: CNN_2 ------------------------------
def make_cls_loaders_for_trial(trial):
    """Rebuild CNN classifier loaders with tunable batch size if datasets are available."""
    if not HAS_DATASETS:
        # reuse existing loaders (batch size fixed)
        return cnn_train_loader, cnn_val_loader, cnn_test_loader, None
    bs = trial.suggest_categorical("batch_size", [32, 64, 128])
    train_loader = DataLoader(cnn_train_ds, batch_size=bs, shuffle=True,  num_workers=0,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(cnn_val_ds,   batch_size=bs, shuffle=False, num_workers=0,
                              pin_memory=True, drop_last=False)
    test_loader  = DataLoader(cnn_test_ds,  batch_size=bs, shuffle=False, num_workers=0,
                              pin_memory=True, drop_last=False)
    return train_loader, val_loader, test_loader, bs

def objective_cnn2_classifier(trial):
    # Model / opt params to search
    proj_dim = trial.suggest_categorical("proj_dim", [128, 256, 384])
    l2norm   = trial.suggest_categorical("l2norm",  [True, False])
    epochs   = trial.suggest_int("epochs", 5, 50)
    lr       = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    wd       = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    opt_name = trial.suggest_categorical("optimizer", ["adam", "adamw"])

    train_loader, val_loader, _test_loader, _bs = make_cls_loaders_for_trial(trial)

    num_classes = len(MAC_ID_LIST)
    model = CNN_2(num_classes=num_classes, proj_dim=proj_dim, l2norm=l2norm).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = (optim.Adam if opt_name=="adam" else optim.AdamW)(model.parameters(), lr=lr, weight_decay=wd)

    # reuse your training loop (validates each epoch on val_loader via test())
    train(model=model, train_loader=train_loader, test_loader=val_loader,
          criterion=criterion, optimizer=optimizer, num_epochs=epochs)

    # metric for Optuna: val@1 in [0,1]
    val_pct = test(model, val_loader)
    return val_pct / 100.0

def run_study_cnn2(study_name="CNN2_CLASSIFIER", n_trials=25, seed=SEED):
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    pruner  = MedianPruner(n_startup_trials=max(5, min(10, n_trials//3)))
    study   = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=study_name)
    study.optimize(objective_cnn2_classifier, n_trials=n_trials, gc_after_trial=True)
    print(f"[{study_name}] Best val@1 = {study.best_value*100:.2f}%")
    print(f"[{study_name}] Best params = {study.best_trial.params}")
    return study

def retrain_and_test_cnn2(best_params, study_name="CNN2_CLASSIFIER"):
    # Rebuild loaders with best batch size if we have datasets
    if HAS_DATASETS:
        bs = best_params.get("batch_size", 64)
        train_loader = DataLoader(cnn_train_ds, batch_size=bs, shuffle=True,  num_workers=0,
                                  pin_memory=True, drop_last=True)
        val_loader   = DataLoader(cnn_val_ds,   batch_size=bs, shuffle=False, num_workers=0,
                                  pin_memory=True, drop_last=False)
        test_loader  = DataLoader(cnn_test_ds,  batch_size=bs, shuffle=False, num_workers=0,
                                  pin_memory=True, drop_last=False)
    else:
        train_loader, val_loader, test_loader = cnn_train_loader, cnn_val_loader, cnn_test_loader

    proj_dim = best_params.get("proj_dim", 256)
    l2norm   = best_params.get("l2norm", True)
    epochs   = max(best_params.get("epochs", 10), 10)
    lr       = best_params.get("lr", 1e-3)
    wd       = best_params.get("weight_decay", 1e-4)
    opt_name = best_params.get("optimizer", "adamw")

    num_classes = len(MAC_ID_LIST)
    model = CNN_2(num_classes=num_classes, proj_dim=proj_dim, l2norm=l2norm).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = (optim.Adam if opt_name=="adam" else optim.AdamW)(model.parameters(), lr=lr, weight_decay=wd)

    train(model=model, train_loader=train_loader, test_loader=val_loader,
          criterion=criterion, optimizer=optimizer, num_epochs=epochs)

    test_pct = test(model, test_loader)
    print(f"[CNN_2 BEST] TEST top-1: {test_pct:.2f}%")
    # save checkpoint
    save_checkpoint(model, study_name)
    return test_pct / 100.0

# ============================ study_5: ResNet50CSI classifier ============================
def objective_resnet50csi_classifier(trial):
    """HPO objective for the supervised ResNet50CSI classifier (not CLIP)."""
    # Hyperparams
    epochs   = trial.suggest_int("epochs", 1, 12)
    lr       = trial.suggest_float("lr", 1e-6, 3e-4, log=True)   # ResNet usually likes smaller LR
    wd       = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    opt_name = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    pretrained = trial.suggest_categorical("pretrained", [True, False])

    # Optional batch size tuning (only if datasets are available)
    if HAS_DATASETS:
        bs = trial.suggest_categorical("batch_size", [32, 64, 128])
        train_loader = DataLoader(cnn_train_ds, batch_size=bs, shuffle=True,  num_workers=0,
                                  pin_memory=True, drop_last=True)
        val_loader   = DataLoader(cnn_val_ds,   batch_size=bs, shuffle=False, num_workers=0,
                                  pin_memory=True, drop_last=False)
    else:
        train_loader, val_loader = cnn_train_loader, cnn_val_loader

    # Model
    num_classes = len(MAC_ID_LIST)
    model = ResNet50CSI(num_classes=num_classes, pretrained=pretrained).to(device)

    # Loss/opt
    criterion = nn.CrossEntropyLoss()
    optimizer = (optim.Adam if opt_name == "adam" else optim.AdamW)(model.parameters(), lr=lr, weight_decay=wd)

    # Train on TRAIN, validate on VAL (reusing your loops)
    train(model=model,
          train_loader=train_loader,
          test_loader=val_loader,   # your train() uses this as validation each epoch
          criterion=criterion,
          optimizer=optimizer,
          num_epochs=epochs)

    # Objective = final VAL accuracy (0..1)
    val_pct = test(model, val_loader)
    return val_pct / 100.0


def run_study_resnet50csi(study_name="RESNET50CSI_CLASSIFIER", n_trials=25, seed=SEED):
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    pruner  = MedianPruner(n_startup_trials=max(5, min(10, n_trials//3)))
    study   = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=study_name)
    study.optimize(objective_resnet50csi_classifier, n_trials=n_trials, gc_after_trial=True)
    print(f"[{study_name}] Best val@1 = {study.best_value*100:.2f}%")
    print(f"[{study_name}] Best params = {study.best_trial.params}")
    return study


def retrain_and_test_resnet50csi(best_params, study_name="RESNET50CSI_CLASSIFIER"):
    """Retrain ResNet50CSI with best params, then evaluate on TEST; save ckpt."""
    # Rebuild loaders with best batch size if datasets available
    if HAS_DATASETS:
        bs = best_params.get("batch_size", 64)
        train_loader = DataLoader(cnn_train_ds, batch_size=bs, shuffle=True,  num_workers=0,
                                  pin_memory=True, drop_last=True)
        val_loader   = DataLoader(cnn_val_ds,   batch_size=bs, shuffle=False, num_workers=0,
                                  pin_memory=True, drop_last=False)
        test_loader  = DataLoader(cnn_test_ds,  batch_size=bs, shuffle=False, num_workers=0,
                                  pin_memory=True, drop_last=False)
    else:
        train_loader, val_loader, test_loader = cnn_train_loader, cnn_val_loader, cnn_test_loader

    # Pull best hyperparams
    epochs    = max(best_params.get("epochs", 6), 6)  # bump a bit for final fit
    lr        = best_params.get("lr", 1e-4)
    wd        = best_params.get("weight_decay", 1e-4)
    opt_name  = best_params.get("optimizer", "adamw")
    pretrained = best_params.get("pretrained", True)

    # Model/opt
    num_classes = len(MAC_ID_LIST)
    model = ResNet50CSI(num_classes=num_classes, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = (optim.Adam if opt_name == "adam" else optim.AdamW)(model.parameters(), lr=lr, weight_decay=wd)

    # Retrain then test
    train(model=model,
          train_loader=train_loader,
          test_loader=val_loader,
          criterion=criterion,
          optimizer=optimizer,
          num_epochs=epochs)

    test_pct = test(model, test_loader)
    print(f"[ResNet50CSI BEST] TEST top-1: {test_pct:.2f}%")

    # Save checkpoint
    save_checkpoint(model, study_name)
    return test_pct / 100.0
# ============================================================================ 



# ------------------------------ Helper: CLIP ------------------------------
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
    if is_resnet:
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    else:
        lr = trial.suggest_float("lr", 5e-5, 3e-3, log=True)
    wd       = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    proj_dim = trial.suggest_categorical("proj_dim", [128, 256, 384])
    return batch_size, epochs, lr, wd, proj_dim

def build_cnn_encoder(proj_dim):
    return CSIEncoder(in_ch=2, proj_dim=proj_dim)

def build_resnet_encoder(proj_dim, trial):
    freeze_bn = trial.suggest_categorical("freeze_backbone_bn", [False, True])
    return CSIResNet50Encoder(proj_dim=proj_dim, pretrained=True, freeze_backbone_bn=freeze_bn)

# ---- study_1: CLIP (CNN, MAC hex→learnable) ----
def objective_clip_cnn_mac(trial):
    batch_size, epochs, lr, wd, proj_dim = suggest_common_hparams(trial, is_resnet=False)
    hex_dim = trial.suggest_categorical("hex_dim", [32, 64, 128])

    label_side = LabelHexProjector(MAC_ID_LIST, dim=proj_dim, hex_dim=hex_dim)
    csi_side   = build_cnn_encoder(proj_dim)
    model      = CSI_CLIP(csi_encoder=csi_side, label_encoder=label_side).to(device)

    _, val_acc = train_clip_once(model, clip_train_ds=clip_train_ds, clip_val_ds=clip_val_ds,
                                 epochs=epochs, batch_size=batch_size, lr=lr, wd=wd,
                                 device=("cuda" if torch.cuda.is_available() else "cpu"))
    return val_acc

# ---- study_2: CLIP (CNN, MAC hex→learnable + LOC RFF) ----
def objective_clip_cnn_mac_loc(trial):
    batch_size, epochs, lr, wd, proj_dim = suggest_common_hparams(trial, is_resnet=False)
    hex_dim    = trial.suggest_categorical("hex_dim", [32, 64, 128])
    lambda_pos = trial.suggest_float("lambda_pos", 0.1, 3.0, log=True)
    sigma      = trial.suggest_float("rff_sigma", 0.5, 3.0, log=True)
    coord_s    = trial.suggest_float("coord_scale", 0.5, 3.0, log=True)

    label_side = LabelHexPlusLoc(MAC_ID_LIST, mac_to_xyz, dim=proj_dim, hex_dim=hex_dim,
                                 lambda_pos=lambda_pos, sigma=sigma, seed=SEED, coord_scale=coord_s)
    csi_side   = build_cnn_encoder(proj_dim)
    model      = CSI_CLIP(csi_encoder=csi_side, label_encoder=label_side).to(device)

    _, val_acc = train_clip_once(model, clip_train_ds=clip_train_ds, clip_val_ds=clip_val_ds,
                                 epochs=epochs, batch_size=batch_size, lr=lr, wd=wd,
                                 device=("cuda" if torch.cuda.is_available() else "cpu"))
    return val_acc

# ---- study_3: CLIP (ResNet50, MAC hex→learnable) ----
def objective_clip_resnet_mac(trial):
    batch_size, epochs, lr, wd, proj_dim = suggest_common_hparams(trial, is_resnet=True)
    hex_dim = trial.suggest_categorical("hex_dim", [32, 64, 128])

    label_side = LabelHexProjector(MAC_ID_LIST, dim=proj_dim, hex_dim=hex_dim)
    csi_side   = build_resnet_encoder(proj_dim, trial)
    model      = CSI_CLIP(csi_encoder=csi_side, label_encoder=label_side).to(device)

    _, val_acc = train_clip_once(model, clip_train_ds=clip_train_ds, clip_val_ds=clip_val_ds,
                                 epochs=epochs, batch_size=batch_size, lr=lr, wd=wd,
                                 device=("cuda" if torch.cuda.is_available() else "cpu"))
    return val_acc

# ---- study_4: CLIP (ResNet50, MAC hex→learnable + LOC RFF) ----
def objective_clip_resnet_mac_loc(trial):
    batch_size, epochs, lr, wd, proj_dim = suggest_common_hparams(trial, is_resnet=True)
    hex_dim    = trial.suggest_categorical("hex_dim", [32, 64, 128])
    lambda_pos = trial.suggest_float("lambda_pos", 0.1, 3.0, log=True)
    sigma      = trial.suggest_float("rff_sigma", 0.5, 3.0, log=True)
    coord_s    = trial.suggest_float("coord_scale", 0.5, 3.0, log=True)

    label_side = LabelHexPlusLoc(MAC_ID_LIST, mac_to_xyz, dim=proj_dim, hex_dim=hex_dim,
                                 lambda_pos=lambda_pos, sigma=sigma, seed=SEED, coord_scale=coord_s)
    csi_side   = build_resnet_encoder(proj_dim, trial)
    model      = CSI_CLIP(csi_encoder=csi_side, label_encoder=label_side).to(device)

    _, val_acc = train_clip_once(model, clip_train_ds=clip_train_ds, clip_val_ds=clip_val_ds,
                                 epochs=epochs, batch_size=batch_size, lr=lr, wd=wd,
                                 device=("cuda" if torch.cuda.is_available() else "cpu"))
    return val_acc

# ---------------------- shared study runner & final eval ----------------------
def run_study(objective_fn, study_name: str, n_trials: int, seed: int = SEED):
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    pruner  = MedianPruner(n_startup_trials=max(5, min(10, n_trials//3)))
    study   = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=study_name)
    study.optimize(objective_fn, n_trials=n_trials, gc_after_trial=True)
    print(f"[{study_name}] Best val@1 = {study.best_value*100:.2f}%")
    print(f"[{study_name}] Best params = {study.best_trial.params}")
    return study

def retrain_and_test(best_params, *, is_resnet: bool, use_location: bool, study_name: str):
    batch_size = best_params["batch_size"]
    epochs     = max(best_params["epochs"], 10)  # bump for final fit
    lr         = best_params["lr"]
    wd         = best_params["weight_decay"]
    proj_dim   = best_params["proj_dim"]

    if use_location:
        hex_dim    = best_params["hex_dim"]
        lambda_pos = best_params["lambda_pos"]
        sigma      = best_params["rff_sigma"]
        coord_s    = best_params["coord_scale"]
        label_side = LabelHexPlusLoc(MAC_ID_LIST, mac_to_xyz, dim=proj_dim, hex_dim=hex_dim,
                                     lambda_pos=lambda_pos, sigma=sigma, seed=SEED, coord_scale=coord_s)
    else:
        label_side = LabelHexProjector(MAC_ID_LIST, dim=proj_dim, hex_dim=best_params["hex_dim"])

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
    # save checkpoint
    save_checkpoint(model, study_name)
    return test_acc
    return test_acc
