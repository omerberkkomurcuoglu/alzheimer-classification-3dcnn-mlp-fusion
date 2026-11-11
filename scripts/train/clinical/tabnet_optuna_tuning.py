"""
13.4.0 - Optuna TabNet Tuning
Optimizes TabNet hyper-parameters using Composite Score.
"""

import os
import random
import warnings
import json
import optuna
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import QuantileTransformer
from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings("ignore")

CFG = {
    "seed": 42,
    "num_classes": 3,
    "csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\0-data_leakage_optimized.csv",
    "save_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\weights",
    "n_trials": 30, # TabNet is much slower than XGB/RF, so we do fewer trials
    "selection_bacc_weight": 0.6,
    "selection_acc_weight": 0.4,
}

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def encode_labels(df: pd.DataFrame) -> np.ndarray:
    label_map = {0: 0, 0.5: 1, 1: 2, 2: 2}
    labels = []
    for v in df["memory"]:
        try: v_f = float(v)
        except: v_f = 0.0
        labels.append(label_map.get(v_f, 0))
    return np.array(labels, dtype=np.int64)

def objective(trial: optuna.Trial) -> float:
    data_df = pd.read_csv(CFG["csv_path"])
    clinical_cols = [
        col for col in data_df.columns 
        if col not in ['OASISID', 'memory', 'path'] and not str(col).startswith('Unnamed:')
    ]
    
    train_df, val_df, _ = make_group_splits(data_df)
    
    for col in clinical_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        val_df[col] = pd.to_numeric(val_df[col], errors='coerce').fillna(0)
    
    scaler = QuantileTransformer(output_distribution='normal', random_state=CFG["seed"])
    X_train = scaler.fit_transform(train_df[clinical_cols])
    y_train = encode_labels(train_df)
    X_val = scaler.transform(val_df[clinical_cols])
    y_val = encode_labels(val_df)
    
    # -------------------------
    # Hyperparameter Search Space
    # -------------------------
    n_da = trial.suggest_int("n_da", 8, 64, step=8)
    param = {
        "n_d": n_da,
        "n_a": n_da,
        "n_steps": trial.suggest_int("n_steps", 3, 7),
        "gamma": trial.suggest_float("gamma", 1.0, 1.8),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-5, 1e-2, log=True),
        "n_independent": trial.suggest_int("n_independent", 1, 4),
        "n_shared": trial.suggest_int("n_shared", 1, 4),
        "optimizer_lr": trial.suggest_float("optimizer_lr", 1e-3, 3e-2, log=True),
    }

    model = TabNetClassifier(
        n_d=param["n_d"], n_a=param["n_a"], n_steps=param["n_steps"],
        gamma=param["gamma"], lambda_sparse=param["lambda_sparse"],
        n_independent=param["n_independent"], n_shared=param["n_shared"],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=param["optimizer_lr"], weight_decay=1e-4),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 20, "gamma": 0.7},
        mask_type='entmax',
        device_name='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=0
    )
    
    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['balanced_accuracy'],
        max_epochs=80, # Fewer epochs for tuning
        patience=15,
        batch_size=256,
        virtual_batch_size=128,
        num_workers=0,
        weights=1,
        drop_last=False
    )
    
    val_preds = model.predict(X_val)
    acc = accuracy_score(y_val, val_preds) * 100.0
    bacc = balanced_accuracy_score(y_val, val_preds) * 100.0
    
    composite_score = (CFG["selection_bacc_weight"] * bacc) + (CFG["selection_acc_weight"] * acc)
    return composite_score

def main() -> None:
    seed_everything(CFG["seed"])
    os.makedirs(CFG["save_dir"], exist_ok=True)
    
    print("Starting Optuna Study for TabNet...")
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///optuna_tabnet.db",
        study_name="tabnet_clinical_tuning",
        load_if_exists=True,
    )
    
    study.optimize(objective, n_trials=CFG["n_trials"])
    
    print("\nBest Trial:")
    print(f"  Value (Composite Score): {study.best_value:.2f}%")
    print("  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
        
    best_params_path = os.path.join(CFG["save_dir"], "optuna_tabnet_best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\nBest parameters saved to {best_params_path}")

if __name__ == "__main__":
    main()
