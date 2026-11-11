"""
8.1 - Train MLP (Clinical) - NEW STABLE VERSION

Trains the DynamicMLP using the best hyperparameters found by 8.0-optuna_mlp_again.py.
Applies EMAModel, Mixed Precision (AMP), and standard CrossEntropyLoss.
"""

import os
import random
import time
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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
    "optuna_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\weights\optuna_mlp_again",
    "save_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\weights\MLP_Again_Stable_best.pth",
    "epochs": 150,
    "patience": 30,
    "grad_accum_steps": 2,
    "ema_decay": 0.998,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# DATA SPLITTING
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


def coerce_label(v):
    try: v = float(v)
    except: v = 0.0
    return {0: 0, 0.5: 1, 1: 2, 2: 2}.get(v, 0)


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


def main():
    seed_everything(CFG["seed"])
    print(f"Device: {DEVICE}")

    # Load Optuna Best Params
    best_params_path = r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\weights\8-optuna_mlp.json"
    if not os.path.exists(best_params_path):
        print(f"Error: {best_params_path} not found. Extract params first.")
        return
        
    with open(best_params_path, "r") as f:
        hp_data = json.load(f)
        hp = hp_data["params"]
        
    print("Loaded Hyperparameters:")
    for k, v in hp.items():
        print(f"  {k}: {v}")

    # Prepare Data
    data_csv = pd.read_csv(CFG["csv_path"])
    clinical_cols = [
        c for c in data_csv.columns
        if c not in ["OASISID", "memory", "path"] + DROP_COLS
        and not str(c).startswith("Unnamed:")
    ]
    train_df, val_df, test_df = make_group_splits(data_csv)

    for col in clinical_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0)
        val_df[col] = pd.to_numeric(val_df[col], errors="coerce").fillna(0)
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce").fillna(0)

    scaler = QuantileTransformer(output_distribution="normal", random_state=CFG["seed"])
    train_df[clinical_cols] = scaler.fit_transform(train_df[clinical_cols])
    val_df[clinical_cols] = scaler.transform(val_df[clinical_cols])
    test_df[clinical_cols] = scaler.transform(test_df[clinical_cols])

    train_ds = ClinicalDataset(train_df, clinical_cols, augment=True)
    val_ds = ClinicalDataset(val_df, clinical_cols, augment=False)
    test_ds = ClinicalDataset(test_df, clinical_cols, augment=False)

    train_counts = np.bincount(np.array([coerce_label(v) for v in train_df["memory"]], dtype=np.int64), minlength=CFG["num_classes"])
    
    # Loaders
    targets = np.array([coerce_label(v) for v in train_ds.data_csv["memory"]], dtype=np.int64)
    weights = 1.0 / np.power(train_counts.astype(np.float64) + 1e-8, hp.get("sampler_power", 0.75))
    sampler = WeightedRandomSampler(torch.as_tensor(weights[targets], dtype=torch.double), len(targets), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=hp.get("batch_size", 64), sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    layers_config = []
    n_layers = hp.get("n_layers", 3)
    for i in range(n_layers):
        layers_config.append(hp.get(f"units_l{i}", 128))
        
    model = DynamicMLP(len(clinical_cols), CFG["num_classes"], layers_config,
                       hp.get("dropout", 0.3), hp.get("use_residual", True)).to(DEVICE)
                       
    use_ema = hp.get("use_ema", True)
    ema = EMAModel(model, CFG["ema_decay"]) if use_ema else None

    criterion = nn.CrossEntropyLoss(label_smoothing=hp.get("label_smoothing", 0.05))
    optimizer = optim.AdamW(model.parameters(), lr=hp.get("lr", 1e-3), weight_decay=hp.get("weight_decay", 1e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=hp.get("scheduler_T0", 10), T_mult=2)
    grad_scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    # Training Loop
    best_bacc = 0.0
    patience_left = CFG["patience"]
    
    os.makedirs(os.path.dirname(CFG["save_path"]), exist_ok=True)

    print("\nStarting Training...")
    for epoch in range(CFG["epochs"]):
        model.train()
        total_loss, total_count = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        
        for step, (features, labels) in enumerate(train_loader):
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            with torch.amp.autocast('cuda', enabled=grad_scaler is not None):
                outputs = model(features)
                loss = criterion(outputs, labels) / CFG["grad_accum_steps"]
                
            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = (step + 1) % CFG["grad_accum_steps"] == 0 or (step + 1) == len(train_loader)
            if should_step:
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                if ema is not None:
                    ema.update(model)
                    
                optimizer.zero_grad(set_to_none=True)
                
            total_loss += loss.item() * CFG["grad_accum_steps"] * labels.size(0)
            total_count += labels.size(0)

        scheduler.step()

        # Validate
        if ema is not None:
            ema.apply_shadow(model)
            
        val_metrics = evaluate(model, val_loader, criterion)
        
        if ema is not None:
            ema.restore(model)
            
        train_loss = total_loss / max(total_count, 1)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]
        val_bacc = val_metrics["bacc"]

        improved = False
        if val_bacc > best_bacc:
            best_bacc = val_bacc
            if ema is not None:
                torch.save(ema.shadow, CFG["save_path"])
            else:
                torch.save(model.state_dict(), CFG["save_path"])
            improved = True
            patience_left = CFG["patience"]
        else:
            patience_left -= 1

        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"Acc: {val_acc:.2f}% BAcc: {val_bacc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"{'🌟' if improved else ''} (Pat: {patience_left})")

        if patience_left <= 0 and epoch >= 20:
            print("Early stopping triggered!")
            break

    print(f"\nTraining completed! Best Validation BAcc: {best_bacc:.2f}%")
    print(f"Model saved at: {CFG['save_path']}")


if __name__ == "__main__":
    main()
