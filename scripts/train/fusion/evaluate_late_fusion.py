"""
Late Fusion Evaluation Pipeline
================================
10 Combinations: 2 MRI (CustomCNN, ResNet50) x 5 Clinical (MLP, XGBoost, TabNet, FT-Transformer, RF)
alpha = 0.4 -> MRI 40%, Clinical 60%

Usage:
    python scripts/train/fusion/fusion_evaluate_all.py
"""

from __future__ import annotations

import os
import sys
import random
import warnings
import json
import joblib

# -- Path Setup --
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(TRAIN_DIR)
sys.path.append(os.path.join(TRAIN_DIR, "mr"))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import QuantileTransformer, label_binarize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mri_pipeline import (
    coerce_memory_label,
    filter_existing_mri_rows,
    load_preprocessed_mri_volume,
)
from custom_model import Custom3DCNN

# ============================================================
# CONFIGURATION
# ============================================================
CFG = {
    "seed": 42,
    "num_classes": 3,
    "class_names": ["Normal", "Mild", "Moderate/Severe"],
    "alpha": 0.35,  # MRI weight (Clinical = 1 - alpha)

    # -- Data Paths --
    "processed_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\data\processed_dataset_128",
    "mri_csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\mr\0-data_leakage_mri_clean_180.csv",
    "clinical_csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\0-data_leakage_optimized.csv",

    # -- MRI Model Weights --
    "customcnn_weights": os.path.join(SCRIPT_DIR, "weights", "customcnn.pth"),
    "resnet50_weights": os.path.join(SCRIPT_DIR, "weights", "resnet.pth"),

    # -- Clinical Model Weights --
    "mlp_weights": os.path.join(SCRIPT_DIR, "weights", "MLP_Again_Stable_best.pth"),
    "mlp_optuna_json": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\weights\8-optuna_mlp.json",
    "xgboost_weights": os.path.join(SCRIPT_DIR, "weights", "xgboost.json"),
    "tabnet_weights": os.path.join(SCRIPT_DIR, "weights", "TabNet_Clinical_Optuna.zip"),
    "ft_transformer_weights": os.path.join(SCRIPT_DIR, "weights", "FT_Transformer_Clinical_best.pth"),
    "rf_weights": os.path.join(SCRIPT_DIR, "weights", "RandomForest_Clinical_Optuna_model.pkl"),
    "lightgbm_weights": os.path.join(SCRIPT_DIR, "weights", "LightGBM_Clinical_Optuna_model.txt"),
    "catboost_weights": os.path.join(SCRIPT_DIR, "weights", "CatBoost_Clinical_Optuna_model.cbm"),

    # -- Model Hyperparams --
    "customcnn_dropout": 0.25,
    "resnet50_dropout": 0.30,
    "mlp_dropout": 0.2929,
    "ft_transformer_dropout": 0.283,
    "ft_transformer_d_token": 64,  # Fixed: Checkpoint expects 64

    # -- Inference Settings --

    
    "batch_size": 4,
    "num_workers": 0,

    # -- Output --
    "results_dir": os.path.join(SCRIPT_DIR, "results2"),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MLP_DROP_COLS = [
    "TAXES", "INRELTO", "BORED", "srtfree", "INLIVWTH", "tmb", "MARISTAT", "AFRAID",
    "INSEX", "ENERGY", "BILLS", "DEP", "BPSYS", "HELPLESS",
    "WRTHLESS", "WEIGHT", "WONDRFUL", "HOPELESS", "DEL", "DROPACT",
]


# ============================================================
# MODELS
# ============================================================

class CustomCNNClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.backbone = Custom3DCNN(num_classes=num_classes, dropout=dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class BrainClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        from monai.networks.nets import resnet50 as monai_r50
        backbone = monai_r50(spatial_dims=3, n_input_channels=1, num_classes=1, shortcut_type="B", bias_downsample=False)
        self.conv1, self.bn1, self.act, self.maxpool = backbone.conv1, backbone.bn1, backbone.act, backbone.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.act(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.head(self.pool(x))

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
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(nn.Linear(d_token, d_token // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_token // 2, num_classes))
    def forward(self, x_cont: torch.Tensor) -> torch.Tensor:
        B = x_cont.shape[0]
        x = x_cont.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        return self.head(self.norm(self.transformer(x)[:, 0, :]))

# ============================================================
# HELPERS
# ============================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_group_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import StratifiedGroupKFold
    labels = np.array([coerce_memory_label(v) for v in df["memory"]], dtype=np.int64)
    groups = df["OASISID"].astype(str).to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG["seed"])
    train_val_idx, test_idx = next(outer.split(df, labels, groups))
    tv_df, test_df = df.iloc[train_val_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=CFG["seed"])
    t_idx, v_idx = next(inner.split(tv_df, labels[train_val_idx], groups[train_val_idx]))
    return tv_df.iloc[t_idx].reset_index(drop=True), tv_df.iloc[v_idx].reset_index(drop=True), test_df

class MRIOnlyDataset(Dataset):
    def __init__(self, root_dir: str, df: pd.DataFrame) -> None:
        self.root_dir, self.df = root_dir, df.reset_index(drop=True)
    def __len__(self) -> int: return len(self.df)
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        vol = load_preprocessed_mri_volume(os.path.join(self.root_dir, f"{row['path']}.npy"))
        return torch.from_numpy(vol).unsqueeze(0), torch.tensor(coerce_memory_label(row["memory"]), dtype=torch.long)

@torch.no_grad()
def predict_mri(model: nn.Module, loader: DataLoader):
    model.eval()
    all_p, all_l = [], []
    for imgs, lbls in tqdm(loader, desc="MRI", leave=False):
        all_p.append(F.softmax(model(imgs.to(DEVICE)), dim=1).cpu().numpy())
        all_l.append(lbls.numpy())
    return np.concatenate(all_p), np.concatenate(all_l)

@torch.no_grad()
def predict_torch(model: nn.Module, X: np.ndarray):
    model.eval()
    return F.softmax(model(torch.tensor(X, dtype=torch.float32).to(DEVICE)), dim=1).cpu().numpy()

def predict_sk(model, X: np.ndarray): return model.predict_proba(X)

# ============================================================
# MAIN
# ============================================================
def main() -> None:
    seed_everything(CFG["seed"])
    os.makedirs(CFG["results_dir"], exist_ok=True)
    
    # 1. Load Data
    mri_csv = filter_existing_mri_rows(CFG["processed_dir"], pd.read_csv(CFG["mri_csv_path"]))
    clin_full = pd.read_csv(CFG["clinical_csv_path"]).drop_duplicates(subset="path", keep="first")
    train_mri, _, test_mri = make_group_splits(mri_csv)
    
    # Alignment
    clin_test = clin_full[clin_full["path"].isin(test_mri["path"])].sort_values("path").reset_index(drop=True)
    clin_train = clin_full[clin_full["path"].isin(train_mri["path"])].reset_index(drop=True)
    mri_test_aligned = test_mri[test_mri["path"].isin(clin_test["path"])].sort_values("path").reset_index(drop=True)
    clin_test = clin_test[clin_test["path"].isin(mri_test_aligned["path"])].reset_index(drop=True)
    test_labels = np.array([coerce_memory_label(v) for v in mri_test_aligned["memory"]], dtype=np.int64)

    # 2. Per-Model Feature Preparation
    cols_full = [c for c in clin_full.columns if c not in ["OASISID", "memory", "path"] and "Unnamed" not in str(c)]
    cols_mlp = [c for c in cols_full if c not in MLP_DROP_COLS]
    
    for c in cols_full:
        clin_train[c] = pd.to_numeric(clin_train[c], errors="coerce").fillna(0)
        clin_test[c] = pd.to_numeric(clin_test[c], errors="coerce").fillna(0)

    # Scaler for Full Features (TabNet, FT-Trans)
    qt_full = QuantileTransformer(output_distribution="normal", random_state=CFG["seed"]).fit(clin_train[cols_full].values)
    X_test_full_qt = qt_full.transform(clin_test[cols_full].values).astype(np.float32)
    X_test_full_raw = clin_test[cols_full].values.astype(np.float32)

    # Scaler for MLP Features (Synchronized fit)
    qt_mlp = QuantileTransformer(output_distribution="normal", random_state=CFG["seed"]).fit(clin_train[cols_mlp].values)
    X_test_mlp_qt = qt_mlp.transform(clin_test[cols_mlp].values).astype(np.float32)

    # 3. Model Inference
    # MRI
    mri_probs = {}
    loader = DataLoader(MRIOnlyDataset(CFG["processed_dir"], mri_test_aligned), batch_size=CFG["batch_size"])
    try:
        m = CustomCNNClassifier(3, CFG["customcnn_dropout"]).to(DEVICE)
        m.load_state_dict(torch.load(CFG["customcnn_weights"]))
        mri_probs["CustomCNN"], _ = predict_mri(m, loader)
        print("✅ CustomCNN done.")
    except Exception as e: print(f"❌ CustomCNN: {e}")
    try:
        m = BrainClassifier(3, CFG["resnet50_dropout"]).to(DEVICE)
        m.load_state_dict(torch.load(CFG["resnet50_weights"]))
        mri_probs["ResNet50"], _ = predict_mri(m, loader)
        print("✅ ResNet50 done.")
    except Exception as e: print(f"❌ ResNet50: {e}")

    # Clinical
    clin_probs = {}
    # MLP
    try:
        with open(CFG["mlp_optuna_json"], "r") as f:
            mlp_hp = json.load(f)["params"]
        layers_config = [mlp_hp.get(f"units_l{i}", 128) for i in range(mlp_hp.get("n_layers", 3))]
        m = DynamicMLP(len(cols_mlp), 3, layers_config, mlp_hp.get("dropout", 0.3), mlp_hp.get("use_residual", True)).to(DEVICE)
        m.load_state_dict(torch.load(CFG["mlp_weights"], map_location=DEVICE))
        clin_probs["MLP"] = predict_torch(m, X_test_mlp_qt)
        print("✅ MLP done.")
    except Exception as e: print(f"❌ MLP: {e}")
    # FT-Transformer
    try:
        m = NativeFTTransformer(len(cols_full), CFG["ft_transformer_d_token"], 8, 3, 3, CFG["ft_transformer_dropout"]).to(DEVICE)
        m.load_state_dict(torch.load(CFG["ft_transformer_weights"]))
        clin_probs["FT-Transformer"] = predict_torch(m, X_test_full_qt)
        print("✅ FT-Transformer done.")
    except Exception as e: print(f"❌ FT-Transformer: {e}")
    # XGBoost
    try:
        import xgboost as xgb
        m = xgb.XGBClassifier(); m.load_model(CFG["xgboost_weights"])
        clin_probs["XGBoost"] = predict_sk(m, X_test_full_raw)
        print("✅ XGBoost done.")
    except Exception as e: print(f"❌ XGBoost: {e}")
    # TabNet
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        m = TabNetClassifier()
        # Direct path with .zip extension as expected by TabNet and used in test_tabnet.py
        m.load_model(CFG["tabnet_weights"])
        clin_probs["TabNet"] = predict_sk(m, X_test_full_qt) # Use QT
        print("✅ TabNet done.")
    except Exception as e: print(f"❌ TabNet: {e}")
    # RF
    try:
        m = joblib.load(CFG["rf_weights"])
        clin_probs["RandomForest"] = predict_sk(m, X_test_full_raw)
        print("✅ RandomForest done.")
    except Exception as e: print(f"❌ RandomForest: {e}")
    # LightGBM
    try:
        import lightgbm as lgb
        m = lgb.Booster(model_file=CFG["lightgbm_weights"])
        clin_probs["LightGBM"] = m.predict(X_test_full_raw)
        print("✅ LightGBM done.")
    except Exception as e: print(f"❌ LightGBM: {e}")
    # CatBoost
    try:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier()
        m.load_model(CFG["catboost_weights"])
        clin_probs["CatBoost"] = predict_sk(m, X_test_full_raw)
        print("✅ CatBoost done.")
    except Exception as e: print(f"❌ CatBoost: {e}")

    # 4. Evaluation
    res = []
    y_test_bin = label_binarize(test_labels, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]

    def get_metrics(y_true, y_pred, y_prob):
        acc = 100 * accuracy_score(y_true, y_pred)
        bacc = 100 * balanced_accuracy_score(y_true, y_pred)
        f1 = 100 * f1_score(y_true, y_pred, average="macro")
        prec = 100 * precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = 100 * recall_score(y_true, y_pred, average="macro", zero_division=0)
        
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        auc_micro = 100 * auc(fpr_micro, tpr_micro)
        
        return acc, bacc, f1, prec, rec, auc_micro

    for n, p in {**mri_probs, **clin_probs}.items():
        pr = p.argmax(1)
        acc, bacc, f1, prec, rec, auc_micro = get_metrics(test_labels, pr, p)
        res.append({"Model": n, "Acc": acc, "BAcc": bacc, "F1": f1, "Precision": prec, "Recall": rec, "MicroAUC": auc_micro})

    for mn, mp in mri_probs.items():
        for cn, cp in clin_probs.items():
            f = CFG["alpha"]*mp + (1-CFG["alpha"])*cp; pr = f.argmax(1)
            acc, bacc, f1, prec, rec, auc_micro = get_metrics(test_labels, pr, f)
            res.append({"Model": f"{mn} + {cn}", "Acc": acc, "BAcc": bacc, "F1": f1, "Precision": prec, "Recall": rec, "MicroAUC": auc_micro})
            
            # Confusion Matrix Plot
            import seaborn as sns
            cm = confusion_matrix(test_labels, pr)
            cm_norm = confusion_matrix(test_labels, pr, normalize='true')
            plt.figure(figsize=(7, 6))
            labels = [f"{v1}\n({v2:.1%})" for v1, v2 in zip(cm.flatten(), cm_norm.flatten())]
            labels = np.asarray(labels).reshape(n_classes, n_classes)
            sns.heatmap(cm_norm, annot=labels, fmt="", cmap='Blues', 
                        xticklabels=CFG["class_names"], yticklabels=CFG["class_names"],
                        cbar_kws={'label': 'Proportion'})
            plt.title(f"{mn} + {cn}\nAcc: {acc:.1f}% | BAcc: {bacc:.1f}%\nF1: {f1:.1f}% | Prec: {prec:.1f}% | Rec: {rec:.1f}%", pad=15)
            plt.ylabel('True Label', fontweight='bold')
            plt.xlabel('Predicted Label', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(CFG["results_dir"], f"fusion_{mn}_{cn}_cm.png"), dpi=150)
            plt.close()

            # ROC Curve Plot
            plt.figure(figsize=(8, 6))
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], f[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Compute micro-average ROC
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), f.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            colors = ['#2196F3', '#FF9800', '#E91E63']
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label=f"{CFG['class_names'][i]} (AUC = {roc_auc[i]:0.3f})")
            
            plt.plot(fpr["micro"], tpr["micro"], color='black', lw=2, linestyle=':', 
                     label=f"Micro-avg (AUC = {roc_auc['micro']:0.3f})")
            plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontweight='bold')
            plt.ylabel('True Positive Rate', fontweight='bold')
            plt.title(f'ROC Curve: {mn} + {cn}\nF1: {f1:.1f}% | BAcc: {bacc:.1f}%', pad=15)
            plt.legend(loc="lower right", frameon=True, shadow=True)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(CFG["results_dir"], f"fusion_{mn}_{cn}_roc.png"), dpi=150)
            plt.close()

    df = pd.DataFrame(res).sort_values("BAcc", ascending=False)
    print("\n" + "="*50 + "\n  FUSION SUMMARY\n" + "="*50 + "\n", df.to_string(index=False))
    df.to_csv(os.path.join(CFG["results_dir"], "fusion_results.csv"), index=False)

if __name__ == "__main__": main()
