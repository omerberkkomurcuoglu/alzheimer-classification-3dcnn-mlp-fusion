"""
13.2.0 - Optuna XGBoost Tuning
Optimizes XGBoost hyperparameters using Composite Score.
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
import xgboost as xgb

warnings.filterwarnings("ignore")

CFG = {
    "seed": 42,
    "num_classes": 3,
    "csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\0-data_leakage_optimized.csv",
    "save_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\weights",
    "n_trials": 100,  # XGBoost is very fast
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

def _compute_sample_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=CFG["num_classes"])
    total = len(labels)
    class_weights = total / (CFG["num_classes"] * np.maximum(counts, 1.0))
    return class_weights[labels]

def objective(trial: optuna.Trial) -> float:
    data_df = pd.read_csv(CFG["csv_path"])
    
    # MLP'de edindiğimiz tecrübe ile en temiz veriyi XGBoost'a da veriyoruz
    drop_cols = [
        
    ]
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
    
    train_weights = _compute_sample_weights(y_train)
    
    # -------------------------
    # Hyperparameter Search Space
    # -------------------------
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }

    model = xgb.XGBClassifier(
        **param,
        objective='multi:softprob',
        num_class=CFG["num_classes"],
        random_state=CFG["seed"],
        tree_method="hist",
    )
    
    # Pruning ve Erken Durdurma optimizasyonları
    try:
        model.fit(
            X_train, y_train,
            sample_weight=train_weights,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    except Exception as e:
        print(f"Trial fail: {e}")
        return 0.0
    
    val_preds = model.predict(X_val)
    acc = accuracy_score(y_val, val_preds) * 100.0
    bacc = balanced_accuracy_score(y_val, val_preds) * 100.0
    
    composite_score = (CFG["selection_bacc_weight"] * bacc) + (CFG["selection_acc_weight"] * acc)
    
    trial.report(composite_score, CFG["n_trials"])
    return composite_score

def main() -> None:
    seed_everything(CFG["seed"])
    os.makedirs(CFG["save_dir"], exist_ok=True)
    
    print("Starting Optuna Study for XGBoost...")
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///optuna_xgboost.db",
        study_name="xgboost_clinical_tuning",
        load_if_exists=True,
    )
    
    study.optimize(objective, n_trials=CFG["n_trials"], n_jobs=2)
    
    print("\nBest Trial:")
    print(f"  Value (Composite Score): {study.best_value:.2f}%")
    print("  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
        
    best_params_path = os.path.join(CFG["save_dir"], "optuna_xgboost_best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\nBest parameters saved to {best_params_path}")

if __name__ == "__main__":
    main()
