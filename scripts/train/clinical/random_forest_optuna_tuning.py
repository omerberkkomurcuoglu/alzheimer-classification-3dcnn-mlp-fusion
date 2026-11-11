"""
13.3.0 - Optuna Random Forest Tuning
Optimizes RandomForest hyper-parameters using Composite Score.
"""

import os
import random
import warnings
import json
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

CFG = {
    "seed": 42,
    "num_classes": 3,
    "csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\0-data_leakage_optimized.csv",
    "save_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\weights",
    "n_trials": 50,  # Random Forest is slightly slower than XGBoost on large estimators
    "selection_bacc_weight": 0.6,
    "selection_acc_weight": 0.4,
}

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

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
    
    # 2.1'deki gibi drop_cols boş tutuluyor (104 özellik)
    drop_cols = []
    clinical_cols = [
        col for col in data_df.columns 
        if col not in ['OASISID', 'memory', 'path'] + drop_cols and not str(col).startswith('Unnamed:')
    ]
    
    train_df, val_df, _ = make_group_splits(data_df)
    
    for col in clinical_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        val_df[col] = pd.to_numeric(val_df[col], errors='coerce').fillna(0)
    
    X_train = train_df[clinical_cols].values
    y_train = encode_labels(train_df)
    X_val = val_df[clinical_cols].values
    y_val = encode_labels(val_df)
    
    # -------------------------
    # Hyperparameter Search Space
    # -------------------------
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=100),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }

    model = RandomForestClassifier(
        **param,
        n_jobs=-1,
        random_state=CFG["seed"]
    )
    
    model.fit(X_train, y_train)
    
    val_preds = model.predict(X_val)
    acc = accuracy_score(y_val, val_preds) * 100.0
    bacc = balanced_accuracy_score(y_val, val_preds) * 100.0
    
    composite_score = (CFG["selection_bacc_weight"] * bacc) + (CFG["selection_acc_weight"] * acc)
    return composite_score

def main() -> None:
    seed_everything(CFG["seed"])
    os.makedirs(CFG["save_dir"], exist_ok=True)
    
    print("Starting Optuna Study for Random Forest...")
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///optuna_rf.db",
        study_name="rf_clinical_tuning",
        load_if_exists=True,
    )
    
    study.optimize(objective, n_trials=CFG["n_trials"])
    
    print("\nBest Trial:")
    print(f"  Value (Composite Score): {study.best_value:.2f}%")
    print("  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
        
    best_params_path = os.path.join(CFG["save_dir"], "optuna_rf_best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\nBest parameters saved to {best_params_path}")

if __name__ == "__main__":
    main()
