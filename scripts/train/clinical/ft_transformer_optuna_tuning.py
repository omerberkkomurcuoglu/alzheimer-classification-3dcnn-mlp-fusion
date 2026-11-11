import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

CFG = {
    "seed": 42,
    "num_classes": 3,
    "class_names": ["Normal", "Mild", "Moderate/Severe"],
    "csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\0-data_leakage_optimized.csv",
    "save_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\weights",
    "selection_bacc_weight": 0.6,
    "selection_acc_weight": 0.4,
}

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- MODEL ARCHITECTURE (Exact same as 5.1) ---
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

def make_group_splits(df):
    mri_csv_path = r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\mr\0-data_leakage_mri_clean_180.csv"
    processed_dir = r"C:\Users\berkk\Desktop\Bitirme_Projesi\data\processed_dataset_128"
    mri_df = pd.read_csv(mri_csv_path)
    existing_paths = [p for p in mri_df["path"] if os.path.exists(os.path.join(processed_dir, f"{p}.npy"))]
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
    inner_labels = np.array([coerce_label(v) for v in mri_train_val["memory"]], dtype=np.int64)
    inner_groups = mri_train_val["OASISID"].astype(str).to_numpy()
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    train_idx, val_idx = next(inner.split(mri_train_val, inner_labels, inner_groups))
    mri_train = mri_train_val.iloc[train_idx]
    mri_val = mri_train_val.iloc[val_idx]
    train_paths, val_paths = set(mri_train["path"]), set(mri_val["path"])
    train_df = df[df["path"].isin(train_paths)].drop_duplicates(subset="path").reset_index(drop=True)
    val_df = df[df["path"].isin(val_paths)].drop_duplicates(subset="path").reset_index(drop=True)
    return train_df, val_df

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_df = pd.read_csv(CFG["csv_path"])
    clinical_cols = [col for col in data_df.columns if col not in ['OASISID', 'memory', 'path'] and not str(col).startswith('Unnamed:')]
    train_df, val_df = make_group_splits(data_df)
    
    scaler = QuantileTransformer(output_distribution='normal', random_state=CFG["seed"])
    train_df[clinical_cols] = scaler.fit_transform(train_df[clinical_cols])
    val_df[clinical_cols] = scaler.transform(val_df[clinical_cols])
    
    train_ds = ClinicalDataset(train_df, clinical_cols)
    val_ds = ClinicalDataset(val_df, clinical_cols)
    
    label_map = {0: 0, 0.5: 1, 1: 2, 2: 2}
    targets = np.array([label_map.get(float(v) if str(v).replace('.','').isdigit() else 0.0, 0) for v in train_df["memory"]])
    counts = np.bincount(targets, minlength=3)
    weights = 1.0 / (counts.astype(np.float64) + 1e-8)
    sampler = WeightedRandomSampler(torch.as_tensor(weights[targets], dtype=torch.double), len(targets), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    
    # --- Tunable Hyperparameters ---
    d_token = trial.suggest_int("d_token", 32, 128, step=32)
    n_heads = trial.suggest_categorical("n_heads", [4, 8])
    n_layers = trial.suggest_int("n_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.05, 0.2)

    model = NativeFTTransformer(
        num_continuous=len(clinical_cols),
        d_token=d_token, n_heads=n_heads, n_layers=n_layers,
        num_classes=3, dropout=dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_bacc = 0.0
    for epoch in range(40): # Faster tuning
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        p, l = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                p.extend(model(features).argmax(1).cpu().numpy())
                l.extend(labels.numpy())
        
        bacc = balanced_accuracy_score(l, p) * 100.0
        acc = accuracy_score(l, p) * 100.0
        score = (CFG["selection_bacc_weight"] * bacc) + (CFG["selection_acc_weight"] * acc)
        best_val_bacc = max(best_val_bacc, score)
        
        trial.report(score, epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        
    return best_val_bacc

if __name__ == "__main__":
    seed_everything(CFG["seed"])
    study = optuna.create_study(direction="maximize", storage="sqlite:///optuna_ft_transformer.db", study_name="ft_transformer_tuning", load_if_exists=True)
    study.optimize(objective, n_trials=30)
    print("Best Params:", study.best_params)
    import json
    with open(os.path.join(CFG["save_dir"], "5-optuna_ft_transformer_best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)
