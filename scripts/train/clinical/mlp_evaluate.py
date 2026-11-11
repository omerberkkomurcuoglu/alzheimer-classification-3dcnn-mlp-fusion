"""
8.2 - Test MLP (Clinical) - NEW STABLE VERSION

Evaluates the trained MLP model from 8.1 on the validation and test sets.
Generates metrics, confusion matrix, ROC curve, classification report,
and feature importance (Permutation Importance).
Saves all plots to test/mlp_new directory.
"""

import os
import random
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc, f1_score
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer, label_binarize
from torch.utils.data import DataLoader, Dataset

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
    "optuna_json": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\weights\8-optuna_mlp.json",
    "weights_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\weights\MLP_Again_Stable_best.pth",
    "out_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\graphs\mlp_new",
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        except: label_val = 0.0
        label = self.label_map.get(label_val, 0)
        clinical_data = pd.to_numeric(row[self.clinical_cols], errors="coerce").fillna(0).values.astype(np.float32)
        return torch.tensor(clinical_data), torch.tensor(label, dtype=torch.long)

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

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    preds_all, labels_all, probs_all = [], [], []
    for features, labels in loader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        outputs = model(features)
        
        probs = F.softmax(outputs, dim=1)
        probs_all.extend(probs.cpu().numpy())
        preds_all.extend(outputs.argmax(1).cpu().numpy().tolist())
        labels_all.extend(labels.cpu().numpy().tolist())
        
    return {
        "acc": 100.0 * accuracy_score(labels_all, preds_all),
        "bacc": 100.0 * balanced_accuracy_score(labels_all, preds_all),
        "macro_f1": 100.0 * f1_score(labels_all, preds_all, average="macro", zero_division=0),
        "preds": preds_all,
        "labels": labels_all,
        "probs": np.array(probs_all),
    }

def plot_roc_curve(labels: list, probs: np.ndarray, title: str, filename: str) -> None:
    y_bin = label_binarize(labels, classes=[0, 1, 2])
    n_classes = y_bin.shape[1]
    
    fpr, tpr, roc_auc = dict(), dict(), dict()
    colors = ['#2196F3', '#FF9800', '#E91E63'] 
    
    if n_classes == 1: return
        
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure(figsize=(8, 6))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{CFG["class_names"][i]} (AUC = {roc_auc[i]:0.3f})')
    
    plt.plot(fpr["micro"], tpr["micro"], color='black', lw=2, linestyle='--', label=f'Micro-avg (AUC = {roc_auc["micro"]:0.3f})')
    plt.plot([0, 1], [0, 1], 'k:', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def print_and_plot_metrics(metrics: dict, prefix: str, title_prefix: str):
    print(f"\n============================================================")
    print(f"{title_prefix.upper()} METRİKLERİ")
    print(f"============================================================")
    print(
        f"Accuracy: {metrics['acc']:.2f}% | Balanced Accuracy: {metrics['bacc']:.2f}% | "
        f"Macro-F1: {metrics['macro_f1']:.2f}%"
    )
    print(classification_report(metrics["labels"], metrics["preds"], target_names=CFG["class_names"], zero_division=0))
    
    cm = confusion_matrix(metrics["labels"], metrics["preds"])
    cm_norm = confusion_matrix(metrics["labels"], metrics["preds"], normalize='true')
    
    plt.figure(figsize=(8, 6))
    labels = [f"{count}\n({pct:.1%})" for count, pct in zip(cm.flatten(), cm_norm.flatten())]
    labels = np.asarray(labels).reshape(3, 3)
    sns.heatmap(cm_norm, annot=labels, fmt="", cmap='Blues', xticklabels=CFG["class_names"], yticklabels=CFG["class_names"])
    plt.title(f"{title_prefix} Confusion Matrix\nAcc: {metrics['acc']:.2f}% | BAcc: {metrics['bacc']:.2f}%")
    plt.ylabel('Gerçek (True)')
    plt.xlabel('Tahmin Edilen (Predicted)')
    plt.tight_layout()
    
    cm_path = os.path.join(CFG["out_dir"], f"mlp_{prefix}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✅ Confusion Matrix kaydedildi: {cm_path}")
    
    roc_path = os.path.join(CFG["out_dir"], f"mlp_{prefix}_roc_curve.png")
    plot_roc_curve(metrics["labels"], metrics["probs"], f"{title_prefix} ROC Curves", roc_path)
    print(f"✅ ROC Eğrisi kaydedildi: {roc_path}\n")

def analyze_feature_importance(model, loader, feature_names, hp):
    model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=hp.get("label_smoothing", 0.05))
    
    print("Veriler toplanıyor (Feature Importance)...")
    all_clinical = []
    all_labels = []
    for clinical, labels in loader:
        all_clinical.append(clinical)
        all_labels.append(labels)
        
    all_clinical = torch.cat(all_clinical).to(DEVICE)
    all_labels = torch.cat(all_labels).to(DEVICE).long()
    
    print("Baseline Loss hesaplanıyor...")
    with torch.no_grad():
        outputs = model(all_clinical)
        baseline_loss = criterion(outputs, all_labels).item()
    
    print(f"Baseline Loss: {baseline_loss:.6f}")
    
    feature_importance = []
    
    for i, feature_name in enumerate(tqdm(feature_names, desc="Analyzing Features")):
        shuffled_clinical = all_clinical.clone()
        perm = torch.randperm(shuffled_clinical.size(0))
        shuffled_clinical[:, i] = shuffled_clinical[perm, i]
        
        with torch.no_grad():
            outputs = model(shuffled_clinical)
            loss = criterion(outputs, all_labels).item()
        
        importance = loss - baseline_loss
        feature_importance.append({
            'Feature': feature_name,
            'Importance': importance
        })
        
    return pd.DataFrame(feature_importance).sort_values(by='Importance', ascending=False)

def main():
    seed_everything(CFG["seed"])
    os.makedirs(CFG["out_dir"], exist_ok=True)
    
    if not os.path.exists(CFG["optuna_json"]):
        print(f"Error: {CFG['optuna_json']} not found.")
        return
        
    with open(CFG["optuna_json"], "r") as f:
        hp_data = json.load(f)
        hp = hp_data["params"]
        
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
    scaler.fit(train_df[clinical_cols])
    val_df[clinical_cols] = scaler.transform(val_df[clinical_cols])
    test_df[clinical_cols] = scaler.transform(test_df[clinical_cols])

    val_ds = ClinicalDataset(val_df, clinical_cols)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    test_ds = ClinicalDataset(test_df, clinical_cols)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    layers_config = []
    n_layers = hp.get("n_layers", 3)
    for i in range(n_layers):
        layers_config.append(hp.get(f"units_l{i}", 128))
        
    model = DynamicMLP(len(clinical_cols), CFG["num_classes"], layers_config,
                       hp.get("dropout", 0.3), hp.get("use_residual", True)).to(DEVICE)
                       
    if not os.path.exists(CFG["weights_path"]):
        print(f"HATA: Ağırlık bulunamadı -> {CFG['weights_path']}")
        return
        
    model.load_state_dict(torch.load(CFG["weights_path"], map_location=DEVICE))
    model.eval()
    print(f"Model yüklendi: {CFG['weights_path']}")

    # Validation
    val_metrics = evaluate(model, val_loader)
    print_and_plot_metrics(val_metrics, "val", "Validation")

    # Test
    test_metrics = evaluate(model, test_loader)
    print_and_plot_metrics(test_metrics, "test", "Test")

    # Feature Importance (Test Set)
    df_importance = analyze_feature_importance(model, test_loader, clinical_cols, hp)
    
    csv_path = os.path.join(CFG["out_dir"], "mlp_feature_importance.csv")
    df_importance.to_csv(csv_path, index=False)
    print(f"\n✅ Feature Importance CSV kaydedildi: {csv_path}")
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=df_importance.head(30), palette='viridis')
    plt.title('Top 30 Feature Importance (Stable MLP)')
    plt.xlabel('Loss Increase (Importance)')
    plt.tight_layout()
    fi_png_path = os.path.join(CFG["out_dir"], "mlp_feature_importance.png")
    plt.savefig(fi_png_path, dpi=150)
    plt.close()
    print(f"✅ Feature Importance Grafiği kaydedildi: {fi_png_path}")

if __name__ == "__main__":
    main()
