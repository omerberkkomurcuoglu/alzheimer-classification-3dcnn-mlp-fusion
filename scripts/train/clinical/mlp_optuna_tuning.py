"""
8.0 - Optuna Hyperparameter Tuning for MLP (Clinical) - NEW STABLE VERSION

This script adapts the robust training structure from 2-train_resnet50_stable.py
into the MLP clinical pipeline. It removes BalancedFocalLoss in favor of standard
CrossEntropyLoss (with label smoothing), and introduces EMAModel and Mixed Precision
(AMP) for maximum stability.
"""

import os
import random
import time
import warnings
import json
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import optuna
    from optuna.storages import RDBStorage
except ImportError:
    print("Optuna is not installed. Please run: pip install optuna sqlalchemy")
    import sys
    sys.exit(1)

warnings.filterwarnings("ignore")

# ============================================================
# FIXED CONFIG
# ============================================================
CFG = {
    "seed": 42,
    "num_classes": 3,
    "class_names": ["Normal", "Mild", "Moderate/Severe"],
    "csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\0-data_leakage_optimized.csv",
    "mri_csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\mr\0-data_leakage_mri_clean_180.csv",
    "processed_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\data\processed_dataset_128",
    "weights_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\weights\optuna_mlp_again",
    "batch_size": 64,
    "epochs_per_trial": 120,
    "patience_per_trial": 25,
    "n_trials": 100,
    "study_name": "mlp_clinical_optuna_v8",
    "db_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\weights\optuna_mlp_again\optuna_mlp_v8.db",
    "grad_accum_steps": 2, # Small accumulation for stability
    "ema_decay": 0.998,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Columns to drop (feature pruning from previous analysis)
DROP_COLS = [
    "TAXES", "INRELTO", "BORED", "srtfree", "INLIVWTH", "tmb", "MARISTAT", "AFRAID",
    "INSEX", "ENERGY", "BILLS", "DEP", "BPSYS", "HELPLESS",
    "WRTHLESS", "WEIGHT", "WONDRFUL", "HOPELESS", "DEL", "DROPACT",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ============================================================
# EMA MODEL
# ============================================================
class EMAModel:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.data.clone()
                continue
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

# ============================================================
# DATA SPLITTING (aligned with MRI splits)
# ============================================================
def make_group_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mri_df = pd.read_csv(CFG["mri_csv_path"])
    existing_paths = [
        p for p in mri_df["path"]
        if os.path.exists(os.path.join(CFG["processed_dir"], f"{p}.npy"))
    ]
    mri_df = mri_df[mri_df["path"].isin(existing_paths)].reset_index(drop=True)

    def coerce_label(v):
        try: v = float(v)
        except: v = 0.0
        return 0 if v == 0.0 else (1 if v == 0.5 else 2)

    labels = np.array([coerce_label(v) for v in mri_df["memory"]], dtype=np.int64)
    groups = mri_df["OASISID"].astype(str).to_numpy()

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_val_idx, test_idx = next(outer.split(mri_df, labels, groups))
    mri_train_val = mri_df.iloc[train_val_idx].reset_index(drop=True)
    mri_test = mri_df.iloc[test_idx].reset_index(drop=True)

    inner_labels = np.array([coerce_label(v) for v in mri_train_val["memory"]], dtype=np.int64)
    inner_groups = mri_train_val["OASISID"].astype(str).to_numpy()
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    train_idx, val_idx = next(inner.split(mri_train_val, inner_labels, inner_groups))

    train_paths = set(mri_train_val.iloc[train_idx]["path"].tolist())
    val_paths = set(mri_train_val.iloc[val_idx]["path"].tolist())
    test_paths = set(mri_test["path"].tolist())

    train_df = df[df["path"].isin(train_paths)].drop_duplicates(subset="path", keep="first").reset_index(drop=True)
    val_df = df[df["path"].isin(val_paths)].drop_duplicates(subset="path", keep="first").reset_index(drop=True)
    test_df = df[df["path"].isin(test_paths)].drop_duplicates(subset="path", keep="first").reset_index(drop=True)

    return train_df, val_df, test_df


# ============================================================
# DATASET
# ============================================================
class ClinicalDataset(Dataset):
    def __init__(self, data_csv: pd.DataFrame, clinical_cols: list[str], augment: bool = False) -> None:
        self.data_csv = data_csv.reset_index(drop=True)
        self.clinical_cols = clinical_cols
        self.augment = augment
        self.label_map = {0: 0, 0.5: 1, 1: 2, 2: 2}

    def __len__(self) -> int:
        return len(self.data_csv)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.data_csv.iloc[idx]
        label_val = row["memory"]
        try: label_val = float(label_val)
        except: label_val = 0.0
        label = self.label_map.get(label_val, 0)
        clinical_data = pd.to_numeric(row[self.clinical_cols], errors="coerce").fillna(0).values.astype(np.float32)
        if self.augment:
            noise = np.random.normal(0, 0.03, clinical_data.shape).astype(np.float32)
            clinical_data = clinical_data + noise
        return torch.tensor(clinical_data), torch.tensor(label, dtype=torch.long)


# ============================================================
# MODEL — Residual MLP
# ============================================================
class ResidualBlock(nn.Module):
    """MLP block with optional residual connection."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.residual(x)


class DynamicMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, layers_config: list[int],
                 dropout: float, use_residual: bool = True) -> None:
        super().__init__()
        blocks = []
        prev_dim = input_dim
        for units in layers_config:
            if use_residual:
                blocks.append(ResidualBlock(prev_dim, units, dropout))
            else:
                blocks.append(nn.Sequential(
                    nn.Linear(prev_dim, units),
                    nn.LayerNorm(units),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ))
            prev_dim = units
        self.backbone = nn.Sequential(*blocks)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


# ============================================================
# HELPERS
# ============================================================
def coerce_label(v):
    try: v = float(v)
    except: v = 0.0
    return {0: 0, 0.5: 1, 1: 2, 2: 2}.get(v, 0)


def count_labels(df: pd.DataFrame) -> np.ndarray:
    labels = np.array([coerce_label(v) for v in df["memory"]], dtype=np.int64)
    return np.bincount(labels, minlength=CFG["num_classes"])


def print_split_stats(name: str, df: pd.DataFrame) -> None:
    counts = count_labels(df)
    total = int(counts.sum())
    parts = [f"{cn}={int(c)} ({100.0*c/max(total,1):.1f}%)" for cn, c in zip(CFG["class_names"], counts)]
    print(f"{name}: {len(df)} samples | " + " | ".join(parts))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> dict:
    model.eval()
    total_loss, total_count = 0.0, 0
    preds_all, labels_all = [], []
    for features, labels in loader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            outputs = model(features)
            loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)
        preds_all.extend(outputs.argmax(1).cpu().numpy().tolist())
        labels_all.extend(labels.cpu().numpy().tolist())
    return {
        "loss": total_loss / max(total_count, 1),
        "acc": 100.0 * accuracy_score(labels_all, preds_all),
        "bacc": 100.0 * balanced_accuracy_score(labels_all, preds_all),
        "preds": preds_all,
        "labels": labels_all,
        "pred_counts": np.bincount(np.asarray(preds_all, dtype=np.int64), minlength=CFG["num_classes"]).tolist(),
    }


# ============================================================
# DATA PREPARATION (called once)
# ============================================================
def prepare_data():
    """Load CSV, split, scale, create datasets — called once."""
    data_csv = pd.read_csv(CFG["csv_path"])
    clinical_cols = [
        c for c in data_csv.columns
        if c not in ["OASISID", "memory", "path"] + DROP_COLS
        and not str(c).startswith("Unnamed:")
    ]
    print(f"Total rows: {len(data_csv)} | Clinical Features: {len(clinical_cols)}")

    train_df, val_df, test_df = make_group_splits(data_csv)

    # Ensure numeric
    for col in clinical_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0)
        val_df[col] = pd.to_numeric(val_df[col], errors="coerce").fillna(0)
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce").fillna(0)

    scaler = QuantileTransformer(output_distribution="normal", random_state=CFG["seed"])
    train_df[clinical_cols] = scaler.fit_transform(train_df[clinical_cols])
    val_df[clinical_cols] = scaler.transform(val_df[clinical_cols])
    test_df[clinical_cols] = scaler.transform(test_df[clinical_cols])

    print_split_stats("Train", train_df)
    print_split_stats("Val", val_df)
    print_split_stats("Test", test_df)

    train_ds = ClinicalDataset(train_df, clinical_cols, augment=True)
    val_ds = ClinicalDataset(val_df, clinical_cols, augment=False)
    test_ds = ClinicalDataset(test_df, clinical_cols, augment=False)

    train_counts = count_labels(train_df)
    return train_ds, val_ds, test_ds, train_counts, clinical_cols


# ============================================================
# OBJECTIVE
# ============================================================
def objective(trial: optuna.Trial, train_ds, val_ds, test_ds, train_counts, clinical_cols) -> float:
    trial_start = time.time()

    # --- Sample hyperparameters ---
    n_layers = trial.suggest_int("n_layers", 2, 5)
    layers_config = []
    for i in range(n_layers):
        layers_config.append(trial.suggest_int(f"units_l{i}", 64, 512, step=32))

    hp = {
        "n_layers": n_layers,
        "layers_config": layers_config,
        "dropout": trial.suggest_float("dropout", 0.15, 0.55),
        "lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15),
        "sampler_power": trial.suggest_float("sampler_power", 0.5, 1.0),
        "use_residual": trial.suggest_categorical("use_residual", [True, False]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "scheduler_T0": trial.suggest_int("scheduler_T0", 5, 20, step=5),
        "use_ema": trial.suggest_categorical("use_ema", [True, False]),
    }

    print(f"\n{'='*60}")
    print(f"TRIAL {trial.number} — Hyperparameters:")
    for k, v in hp.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")

    # --- Build loaders ---
    targets = np.array([coerce_label(v) for v in train_ds.data_csv["memory"]], dtype=np.int64)
    weights = 1.0 / np.power(train_counts.astype(np.float64) + 1e-8, hp["sampler_power"])
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights[targets], dtype=torch.double), len(targets), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    # --- Build model ---
    input_dim = len(clinical_cols)
    model = DynamicMLP(input_dim, CFG["num_classes"], hp["layers_config"],
                       hp["dropout"], hp["use_residual"]).to(DEVICE)
                       
    ema = EMAModel(model, CFG["ema_decay"]) if hp["use_ema"] else None

    # Standard CrossEntropyLoss with Label Smoothing (No BalancedFocalLoss)
    criterion = nn.CrossEntropyLoss(label_smoothing=hp["label_smoothing"])

    optimizer = optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=hp["scheduler_T0"], T_mult=2
    )
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    best_score = 0.0
    best_state_dict = None
    patience_left = CFG["patience_per_trial"]

    for epoch in range(CFG["epochs_per_trial"]):
        # --- Train ---
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, (features, labels) in enumerate(train_loader):
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                outputs = model(features)
                loss = criterion(outputs, labels) / CFG["grad_accum_steps"]
                
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = (step + 1) % CFG["grad_accum_steps"] == 0 or (step + 1) == len(train_loader)
            if should_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                if ema is not None:
                    ema.update(model)
                    
                optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        # --- Validate ---
        if ema is not None:
            ema.apply_shadow(model)
            
        val_metrics = evaluate(model, val_loader, criterion)
        
        if ema is not None:
            ema.restore(model)
            
        bacc = val_metrics["bacc"]
        acc = val_metrics["acc"]
        composite = 0.60 * bacc + 0.40 * acc

        improved = False
        if composite > best_score:
            best_score = composite
            # If using EMA, we want to save the shadow weights
            if ema is not None:
                best_state_dict = {k: v.cpu().clone() for k, v in ema.shadow.items()}
            else:
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            improved = True

        patience_left = CFG["patience_per_trial"] if improved else patience_left - 1

        # --- Log every 10 epochs or on improvement ---
        if (epoch + 1) % 10 == 0 or improved:
            pc = val_metrics["pred_counts"]
            lr_text = f"{optimizer.param_groups[0]['lr']:.2e}"
            print(f"  T{trial.number} Ep {epoch+1:03d} | VaL {val_metrics['loss']:.4f} "
                  f"Acc {acc:5.2f}% BAcc {bacc:5.2f}% Comp {composite:5.2f}% | "
                  f"LR {lr_text} | N:{pc[0]} Mi:{pc[1]} MS:{pc[2]} | "
                  f"best={best_score:.2f}% pat={patience_left}")

        # --- Report & Prune ---
        trial.report(best_score, epoch)
        if trial.should_prune():
            print(f"  ⛔ PRUNED at epoch {epoch+1}")
            _save_trial(trial.number, best_state_dict, best_score, hp)
            raise optuna.exceptions.TrialPruned()

        # --- Early stop ---
        if patience_left <= 0 and epoch >= 20:
            print(f"  ⏹ Early stop at epoch {epoch+1} (patience exhausted)")
            break

    # --- Test evaluation (info only) ---
    trial_elapsed = (time.time() - trial_start) / 60.0
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    if best_state_dict is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state_dict.items()})
        
    test_metrics = evaluate(model, test_loader, criterion)
    test_composite = 0.60 * test_metrics["bacc"] + 0.40 * test_metrics["acc"]

    pc_test = test_metrics["pred_counts"]
    print(f"\n  ✅ Trial {trial.number} done | {trial_elapsed:.1f}m")
    print(f"     Val  Composite: {best_score:.2f}% (Optuna Target)")
    print(f"     Test Composite: {test_composite:.2f}% (Acc={test_metrics['acc']:.2f}% BAcc={test_metrics['bacc']:.2f}%) [INFO ONLY]")
    print(f"     Test Preds: N:{pc_test[0]} Mi:{pc_test[1]} MS:{pc_test[2]}")

    _save_trial(trial.number, best_state_dict, best_score, hp, test_metrics)

    trial.set_user_attr("val_composite", best_score)
    trial.set_user_attr("test_acc", test_metrics["acc"])
    trial.set_user_attr("test_bacc", test_metrics["bacc"])
    trial.set_user_attr("test_composite", test_composite)
    trial.set_user_attr("trial_duration_min", trial_elapsed)

    del model, optimizer, scheduler, criterion, test_loader
    if ema is not None:
        del ema
    torch.cuda.empty_cache()
    gc.collect()

    return best_score


def _save_trial(trial_number, state_dict, best_score, hp, test_metrics=None):
    os.makedirs(CFG["weights_dir"], exist_ok=True)
    if state_dict is not None:
        weight_path = os.path.join(CFG["weights_dir"], f"trial_{trial_number:02d}_best.pth")
        torch.save(state_dict, weight_path)

    meta = {"trial_number": trial_number, "best_score": best_score, "hyperparameters": hp}
    if test_metrics is not None:
        test_composite = 0.60 * test_metrics["bacc"] + 0.40 * test_metrics["acc"]
        meta["test_acc"] = round(test_metrics["acc"], 4)
        meta["test_bacc"] = round(test_metrics["bacc"], 4)
        meta["test_composite"] = round(test_composite, 4)
    meta_path = os.path.join(CFG["weights_dir"], f"trial_{trial_number:02d}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)


# ============================================================
# MAIN
# ============================================================
def main():
    seed_everything(CFG["seed"])
    print(f"Device: {DEVICE}")
    os.makedirs(CFG["weights_dir"], exist_ok=True)

    # --- Load data once ---
    train_ds, val_ds, test_ds, train_counts, clinical_cols = prepare_data()

    # --- Optuna Study (SQLite persistent) ---
    storage = RDBStorage(
        url=f"sqlite:///{CFG['db_path']}",
        engine_kwargs={"connect_args": {"timeout": 30}},
    )
    study = optuna.create_study(
        study_name=CFG["study_name"],
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=30,
            interval_steps=5,
        ),
    )

    # --- Recovery for failed/interrupted trials ---
    failed_trials = [
        t for t in study.trials
        if t.state in (optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.FAIL)
    ]
    for ft in failed_trials:
        print(f"  ↩️ Trial {ft.number} re-enqueued...")
        study.enqueue_trial(ft.params)

    remaining = max(0, CFG["n_trials"] - len(study.trials))
    if remaining == 0:
        print(f"\nAll {CFG['n_trials']} trials already completed!")
    else:
        print(f"\n🚀 Starting {remaining} trials (target: {CFG['n_trials']})...")
        study.optimize(
            lambda trial: objective(trial, train_ds, val_ds, test_ds, train_counts, clinical_cols),
            n_trials=remaining,
            gc_after_trial=True,
        )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("OPTUNA SUMMARY")
    print("=" * 60)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"Total: {len(study.trials)} | Completed: {len(completed)} | Pruned: {len(pruned)}")

    if completed:
        completed.sort(key=lambda t: t.value or 0, reverse=True)
        print("\n--- Top 5 Trials ---")
        for i, t in enumerate(completed[:5]):
            val_c = t.user_attrs.get("val_composite", None)
            test_c = t.user_attrs.get("test_composite", None)
            test_b = t.user_attrs.get("test_bacc", None)
            dur = t.user_attrs.get("trial_duration_min", "?")
            print(f"  #{i+1} Trial {t.number}: Val={val_c:.2f}% Test={test_c:.2f}% (BAcc={test_b:.2f}%) | {dur}m")
            for k, v in t.params.items():
                print(f"       {k}: {v}")

        # Copy best weights
        best_trial = completed[0]
        src = os.path.join(CFG["weights_dir"], f"trial_{best_trial.number:02d}_best.pth")
        dst = os.path.join(CFG["weights_dir"], "best_overall.pth")
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, dst)
            print(f"\n🏆 Best weights copied: {dst}")

        # Save best params
        best_params_path = os.path.join(CFG["weights_dir"], "best_params.json")
        with open(best_params_path, "w", encoding="utf-8") as f:
            json.dump({"trial_number": best_trial.number, "best_score": best_trial.value,
                        "params": best_trial.params}, f, indent=4, ensure_ascii=False)
        print(f"📋 Best params saved: {best_params_path}")
    else:
        print("\n⚠️ No trials completed!")


if __name__ == "__main__":
    main()
