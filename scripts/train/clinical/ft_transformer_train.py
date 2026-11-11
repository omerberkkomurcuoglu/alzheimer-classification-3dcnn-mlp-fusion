"""
13.5 - Feature Tokenizer Transformer (FT-Transformer) Classifier
Builds a native PyTorch FT-Transformer architecture from scratch for Tabular Data!
Provides an extreme Deep Learning counterpart without any external dependencies.
"""

from __future__ import annotations

import os
import random
import time
import warnings

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, auc, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import QuantileTransformer, label_binarize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import json

CFG = {
    "seed": 42,
    "num_classes": 3,
    "class_names": ["Normal", "Mild", "Moderate/Severe"],
    "csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\0-data_leakage_optimized.csv",
    "weights_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\weights",
    "model_name": "FT_Transformer_Clinical",
    
    # Transformer Parameters (Defaults, will be overridden by Optuna JSON if exists)
    "d_token": 64,         
    "n_heads": 8,          
    "n_layers": 3,         
    "dropout": 0.2,        
    
    "batch_size": 128,
    "val_batch_size": 128,
    "test_batch_size": 128,
    "num_workers": 0,
    "epochs": 100,
    "patience": 25,
    "label_smoothing": 0.1,
    "lr": 5e-4,            
    "weight_decay": 1e-4, 
    "use_weighted_sampler": True,
}

# --- Load Optuna Best Params if exist ---
params_path = os.path.join(CFG["weights_dir"], "5-optuna_ft_transformer_best_params.json")
if os.path.exists(params_path):
    print(f"Loading best parameters from {params_path}")
    with open(params_path, "r") as f:
        best_params = json.load(f)
    CFG.update(best_params)
    print("CFG updated with Optuna parameters.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None, gamma: float, label_smoothing: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(-1)
        with torch.no_grad():
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / max(num_classes - 1, 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(inputs, dim=-1)
        probs = log_probs.exp()
        pt = (probs * smooth_targets).sum(dim=-1)
        ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        loss = ((1.0 - pt).pow(self.gamma) * ce_loss)
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()

# =========================================================================
# FT-TRANSFORMER MODEL ARCHITECTURE
# =========================================================================
class NativeFTTransformer(nn.Module):
    def __init__(self, num_continuous: int, d_token: int, n_heads: int, n_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.num_continuous = num_continuous
        
        # Tokenizer for continuous variables
        # We project each of the `num_continuous` scalars into an embedding of size `d_token`
        self.weight = nn.Parameter(torch.Tensor(num_continuous, d_token))
        self.bias = nn.Parameter(torch.Tensor(num_continuous, d_token))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)
        
        # [CLS] Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Transformer Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification Head (processes the CLS token)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_token // 2, num_classes)
        )

    def forward(self, x_cont: torch.Tensor) -> torch.Tensor:
        B = x_cont.shape[0]
        
        # 1. Tokenize Continous -> Embeddings
        # x_cont: (B, num_features)
        # weight: (num_features, d_token)
        # Using broadcasting:
        x = x_cont.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        # x shape: (B, num_features, d_token)
        
        # 2. Add CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, d_token)
        x = torch.cat((cls_tokens, x), dim=1)         # (B, num_features + 1, d_token)
        
        # 3. Transformer Attention Engine
        out = self.transformer(x)                     # (B, num_features + 1, d_token)
        
        # 4. Extract CLS Token and Classify
        cls_out = self.norm(out[:, 0, :])             # (B, d_token)
        return self.head(cls_out)

class ClinicalDataset(Dataset):
    def __init__(self, data_csv: pd.DataFrame, clinical_cols: list[str]) -> None:
        self.data_csv = data_csv.reset_index(drop=True)
        self.clinical_cols = clinical_cols
        self.label_map = {0: 0, 0.5: 1, 1: 2, 2: 2}

    def __len__(self) -> int:
        return len(self.data_csv)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.data_csv.iloc[idx]
        
        label_val = row["memory"]
        try: label_val = float(label_val)
        except Exception: label_val = 0.0
        label = self.label_map.get(label_val, 0)
        
        clinical_series = row[self.clinical_cols]
        clinical_data = pd.to_numeric(clinical_series, errors="coerce").fillna(0).values.astype(np.float32)

        return torch.tensor(clinical_data), torch.tensor(label, dtype=torch.long)

def make_group_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # --- ALIGN WITH MRI DATASET ---
    import os
    mri_csv_path = r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\mr\0-data_leakage_mri_clean_180.csv"
    processed_dir = r"C:\Users\berkk\Desktop\Bitirme_Projesi\data\processed_dataset_128"
    
    mri_df = pd.read_csv(mri_csv_path)
    existing_paths = []
    for p in mri_df["path"]:
        if os.path.exists(os.path.join(processed_dir, f"{p}.npy")):
            existing_paths.append(p)
    mri_df = mri_df[mri_df["path"].isin(existing_paths)].reset_index(drop=True)
    
    def coerce_label(v):
        try: v = float(v)
        except: v = 0.0
        return 0 if v == 0.0 else (1 if v == 0.5 else 2)
        
    labels = np.array([coerce_label(v) for v in mri_df["memory"]], dtype=np.int64)
    groups = mri_df["OASISID"].astype(str).to_numpy()
    
    from sklearn.model_selection import StratifiedGroupKFold
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_val_idx, test_idx = next(outer.split(mri_df, labels, groups))
    mri_train_val = mri_df.iloc[train_val_idx].reset_index(drop=True)
    mri_test = mri_df.iloc[test_idx].reset_index(drop=True)
    
    inner_labels = np.array([coerce_label(v) for v in mri_train_val["memory"]], dtype=np.int64)
    inner_groups = mri_train_val["OASISID"].astype(str).to_numpy()
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    train_idx, val_idx = next(inner.split(mri_train_val, inner_labels, inner_groups))
    
    mri_train = mri_train_val.iloc[train_idx].reset_index(drop=True)
    mri_val = mri_train_val.iloc[val_idx].reset_index(drop=True)
    
    train_paths = set(mri_train["path"].tolist())
    val_paths = set(mri_val["path"].tolist())
    test_paths = set(mri_test["path"].tolist())
    
    # Filter clinical DF by paths (keep one row per path exactly to match MRI)
    train_df = df[df["path"].isin(train_paths)].drop_duplicates(subset="path", keep="first").reset_index(drop=True)
    val_df = df[df["path"].isin(val_paths)].drop_duplicates(subset="path", keep="first").reset_index(drop=True)
    test_df = df[df["path"].isin(test_paths)].drop_duplicates(subset="path", keep="first").reset_index(drop=True)
    
    return train_df, val_df, test_df

def count_mapped_labels(df: pd.DataFrame) -> np.ndarray:
    label_map = {0: 0, 0.5: 1, 1: 2, 2: 2}
    labels = []
    for v in df["memory"]:
        try: v_f = float(v)
        except: v_f = 0.0
        labels.append(label_map.get(v_f, 0))
    return np.bincount(np.array(labels, dtype=np.int64), minlength=CFG["num_classes"])

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> dict[str, object]:
    model.eval()
    total_loss, total_count = 0.0, 0
    preds_all, labels_all, probs_all = [], [], []
    for features, labels in loader:
        features, labels = features.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        outputs = model(features)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)
        
        probs = F.softmax(outputs, dim=1)
        probs_all.extend(probs.cpu().numpy())
        preds_all.extend(outputs.argmax(1).cpu().numpy().tolist())
        labels_all.extend(labels.cpu().numpy().tolist())
        
    return {
        "loss": total_loss / max(total_count, 1),
        "acc": 100.0 * accuracy_score(labels_all, preds_all),
        "bacc": 100.0 * balanced_accuracy_score(labels_all, preds_all),
        "macro_f1": 100.0 * f1_score(labels_all, preds_all, average="macro", zero_division=0),
        "preds": preds_all,
        "labels": labels_all,
        "probs": np.array(probs_all),
    }

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, epoch_idx: int) -> tuple[float, float]:
    model.train()
    total_loss, total_count, total_correct = 0.0, 0, 0
    loop = tqdm(loader, desc=f"Ep {epoch_idx + 1}", leave=False)
    
    for features, labels in loop:
        features, labels = features.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        total_correct += (outputs.argmax(1) == labels).sum().item()
        
        loop.set_postfix(
            train_loss=f"{total_loss / max(total_count, 1):.4f}",
            train_acc=f"{100.0 * total_correct / max(total_count, 1):.2f}%"
        )
    return total_loss / max(total_count, 1), 100.0 * total_correct / max(total_count, 1)

def main() -> None:
    seed_everything(CFG["seed"])
    os.makedirs(CFG["weights_dir"], exist_ok=True)
    print(f"Device: {DEVICE}")
    print("=" * 60)
    print(" FT-TRANSFORMER CLINICAL TRAINING (NATIVE PYTORCH) ")
    print("=" * 60)

    data_csv = pd.read_csv(CFG["csv_path"])
    clinical_cols = [col for col in data_csv.columns if col not in ['OASISID', 'memory', 'path'] and not str(col).startswith('Unnamed:')]

    train_df, val_df, test_df = make_group_splits(data_csv)
    
    # Transformers also love Gaussians
    for col in clinical_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        val_df[col] = pd.to_numeric(val_df[col], errors='coerce').fillna(0)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
    
    scaler = QuantileTransformer(output_distribution='normal', random_state=CFG["seed"])
    train_df[clinical_cols] = scaler.fit_transform(train_df[clinical_cols])
    val_df[clinical_cols] = scaler.transform(val_df[clinical_cols])
    test_df[clinical_cols] = scaler.transform(test_df[clinical_cols])

    train_ds = ClinicalDataset(train_df, clinical_cols)
    val_ds = ClinicalDataset(val_df, clinical_cols)
    test_ds = ClinicalDataset(test_df, clinical_cols)

    train_counts = count_mapped_labels(train_ds.data_csv)
    
    loader_kwargs = {"num_workers": CFG["num_workers"], "pin_memory": True}
    if CFG["use_weighted_sampler"]:
        label_map = {0: 0, 0.5: 1, 1: 2, 2: 2}
        targets = []
        for v in train_ds.data_csv["memory"]:
            try: v_f = float(v)
            except: v_f = 0.0
            targets.append(label_map.get(v_f, 0))
        targets = np.array(targets, dtype=np.int64)
        weights = 1.0 / np.power(train_counts.astype(np.float64) + 1e-8, 1.0)
        sampler = WeightedRandomSampler(torch.as_tensor(weights[targets], dtype=torch.double), len(targets), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler, drop_last=False, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, drop_last=False, **loader_kwargs)
    
    val_loader = DataLoader(val_ds, batch_size=CFG["val_batch_size"], shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=CFG["test_batch_size"], shuffle=False, **loader_kwargs)

    model = NativeFTTransformer(
        num_continuous=len(clinical_cols),
        d_token=CFG["d_token"],
        n_heads=CFG["n_heads"],
        n_layers=CFG["n_layers"],
        num_classes=CFG["num_classes"],
        dropout=CFG["dropout"]
    ).to(DEVICE)
    
    # We use a static Gamma=2.0 Focus here for Transformer 
    criterion = BalancedFocalLoss(alpha=None, gamma=2.0, label_smoothing=CFG["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    best_bacc = -float("inf")
    patience_left = CFG["patience"]

    for epoch in range(CFG["epochs"]):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_metrics = evaluate(model, val_loader, criterion)
        
        scheduler.step()

        improved = False
        if val_metrics["bacc"] > best_bacc:
            best_bacc = float(val_metrics["bacc"])
            torch.save(model.state_dict(), os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_best.pth"))
            improved = True
        
        patience_left = CFG["patience"] if improved else patience_left - 1
        elapsed = (time.time() - start_time) / 60.0
        
        print(f"Ep {epoch + 1:03d} | TrL {train_loss:.4f} TrA {train_acc:5.2f}% | "
              f"VaL {val_metrics['loss']:.4f} VaA {val_metrics['acc']:5.2f}% BAcc {val_metrics['bacc']:5.2f}% | "
              f"LR {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}m | pat={patience_left}")

        if patience_left <= 0:
            print("Early stopping triggered.")
            break

    best_path = os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_best.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    
    test_metrics = evaluate(model, test_loader, criterion)
    print("\nTEST RESULTS (FT-TRANSFORMER)")
    print(f"Accuracy: {test_metrics['acc']:.2f}% | Balanced Accuracy: {test_metrics['bacc']:.2f}%")
    print(classification_report(test_metrics["labels"], test_metrics["preds"], target_names=CFG["class_names"], zero_division=0))

    # Confusion matrix with counts and percentages
    cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"])
    cm_norm = confusion_matrix(test_metrics["labels"], test_metrics["preds"], normalize='true')
    
    # Calculate class-specific accuracies
    class_accs = cm.diagonal() / cm.sum(axis=1)
    normal_acc = class_accs[0] * 100.0
    
    plt.figure(figsize=(8, 6))
    labels = [f"{count}\n({pct:.1%})" for count, pct in zip(cm.flatten(), cm_norm.flatten())]
    labels = np.asarray(labels).reshape(3, 3)
    
    sns.heatmap(cm_norm, annot=labels, fmt="", cmap='Blues',
                xticklabels=CFG["class_names"],
                yticklabels=CFG["class_names"])
    
    plt.title(f"{CFG['model_name']} - Test Confusion Matrix\n"
              f"BAcc: {test_metrics['bacc']:.2f}% | Normal Acc: {normal_acc:.2f}%")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_confusion_matrix.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
