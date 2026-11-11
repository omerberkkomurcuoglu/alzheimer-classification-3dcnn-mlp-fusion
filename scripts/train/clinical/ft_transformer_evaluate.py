import os
import random
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer, label_binarize
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ============================================================
# 1. AYARLAR & SEED
# ============================================================
CFG = {
    "seed": 42,
    "num_classes": 3,
    "class_names": ["Normal", "Mild", "Moderate/Severe"],
    "csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\0-data_leakage_optimized.csv",
    "weights_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\weights",
    "graphs_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\graphs\ft_transformer",
    "model_name": "FT_Transformer_Clinical",
    
    # Transformer Parameters (Defaults, will be overridden by JSON)
    "d_token": 64,
    "n_heads": 8,
    "n_layers": 3,
    "dropout": 0.2,
}

import json
params_path = os.path.join(CFG["weights_dir"], "5-optuna_ft_transformer_best_params.json")
if os.path.exists(params_path):
    with open(params_path, "r") as f:
        CFG.update(json.load(f))
    print(f"Beste parametreler yüklendi: {params_path}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed sabitlendi: {seed}")

set_seed(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. MODEL ARCHITECTURE
# ============================================================
class NativeFTTransformer(nn.Module):
    def __init__(self, num_continuous: int, d_token: int, n_heads: int, n_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.num_continuous = num_continuous
        self.weight = nn.Parameter(torch.Tensor(num_continuous, d_token))
        self.bias = nn.Parameter(torch.Tensor(num_continuous, d_token))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_token // 2, num_classes)
        )

    def forward(self, x_cont: torch.Tensor) -> torch.Tensor:
        B = x_cont.shape[0]
        x = x_cont.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        out = self.transformer(x)
        cls_out = self.norm(out[:, 0, :])
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
        except: label_val = 0.0
        label = self.label_map.get(label_val, 0)
        clinical_data = pd.to_numeric(row[self.clinical_cols], errors="coerce").fillna(0).values.astype(np.float32)
        return torch.tensor(clinical_data), torch.tensor(label, dtype=torch.long)

# ============================================================
# 3. YARDIMCI FONKSİYONLAR
# ============================================================
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
    
    train_df = df[df["path"].isin(train_paths)].drop_duplicates(subset="path", keep="first").reset_index(drop=True)
    val_df = df[df["path"].isin(val_paths)].drop_duplicates(subset="path", keep="first").reset_index(drop=True)
    test_df = df[df["path"].isin(test_paths)].drop_duplicates(subset="path", keep="first").reset_index(drop=True)
    
    return train_df, val_df, test_df

def plot_roc_curve(labels: np.ndarray, probs: np.ndarray, title: str, filename: str) -> None:
    y_bin = label_binarize(labels, classes=[0, 1, 2])
    n_classes = y_bin.shape[1]
    fpr, tpr, roc_auc = dict(), dict(), dict()
    colors = ['#2196F3', '#FF9800', '#E91E63'] 
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

def print_and_plot_metrics(y_true, y_pred, y_prob, prefix: str, title_prefix: str):
    acc = accuracy_score(y_true, y_pred) * 100.0
    bacc = balanced_accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro") * 100.0
    
    print(f"\n============================================================")
    print(f"{title_prefix.upper()} METRİKLERİ")
    print(f"============================================================")
    print(f"Accuracy: {acc:.2f}% | Balanced Accuracy: {bacc:.2f}% | Macro-F1: {macro_f1:.2f}%")
    print(classification_report(y_true, y_pred, target_names=CFG["class_names"], zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(8, 6))
    labels = [f"{count}\n({pct:.1%})" for count, pct in zip(cm.flatten(), cm_norm.flatten())]
    labels = np.asarray(labels).reshape(3, 3)
    sns.heatmap(cm_norm, annot=labels, fmt="", cmap='Purples', xticklabels=CFG["class_names"], yticklabels=CFG["class_names"])
    plt.title(f"{CFG['model_name']} - {title_prefix} Confusion Matrix\nAccuracy: {acc:.2f}% | BAcc: {bacc:.2f}%")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    cm_path = os.path.join(CFG["graphs_dir"], f"transformer_{prefix}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✅ Confusion Matrix kaydedildi: {cm_path}")
    
    roc_path = os.path.join(CFG["graphs_dir"], f"transformer_{prefix}_roc_curve.png")
    plot_roc_curve(y_true, y_prob, f"{title_prefix} ROC Curves (FT-Transformer)", roc_path)
    print(f"✅ ROC Eğrisi kaydedildi: {roc_path}")

# ============================================================
# 4. MAIN
# ============================================================
def main():
    os.makedirs(CFG["graphs_dir"], exist_ok=True)
    data_df = pd.read_csv(CFG["csv_path"])
    clinical_cols = [col for col in data_df.columns if col not in ['OASISID', 'memory', 'path'] and not str(col).startswith('Unnamed:')]
    
    train_df, val_df, test_df = make_group_splits(data_df)
    
    for col in clinical_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        val_df[col] = pd.to_numeric(val_df[col], errors='coerce').fillna(0)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
    
    scaler = QuantileTransformer(output_distribution='normal', random_state=CFG["seed"])
    scaler.fit(train_df[clinical_cols])
    val_df[clinical_cols] = scaler.transform(val_df[clinical_cols])
    test_df[clinical_cols] = scaler.transform(test_df[clinical_cols])
    
    val_ds = ClinicalDataset(val_df, clinical_cols)
    test_ds = ClinicalDataset(test_df, clinical_cols)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    model = NativeFTTransformer(
        num_continuous=len(clinical_cols),
        d_token=CFG["d_token"], n_heads=CFG["n_heads"], n_layers=CFG["n_layers"],
        num_classes=3, dropout=CFG["dropout"]
    ).to(DEVICE)

    weight_path = os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_best.pth")
    if not os.path.exists(weight_path):
        print(f"HATA: Model ağırlığı bulunamadı: {weight_path}")
        return
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()
    print(f"Model yüklendi: {weight_path}")

    # Eval
    for loader, prefix, title in [(val_loader, "val", "Validation"), (test_loader, "test", "Test")]:
        preds_all, labels_all, probs_all = [], [], []
        with torch.no_grad():
            for features, labels in loader:
                features = features.to(DEVICE)
                outputs = model(features)
                probs = F.softmax(outputs, dim=1)
                probs_all.extend(probs.cpu().numpy())
                preds_all.extend(outputs.argmax(1).cpu().numpy().tolist())
                labels_all.extend(labels.numpy().tolist())
        print_and_plot_metrics(np.array(labels_all), np.array(preds_all), np.array(probs_all), prefix, title)

if __name__ == "__main__":
    main()
