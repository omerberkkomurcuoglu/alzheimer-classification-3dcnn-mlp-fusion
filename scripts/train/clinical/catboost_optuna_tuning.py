"""
7.0 - Optuna CatBoost Tuning
Optimizes CatBoost hyperparameters using Composite Score (60% BAcc + 40% Acc).
"""

import os
import random
import warnings
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from catboost import CatBoostClassifier, Pool

try:
    import optuna
    from optuna.storages import RDBStorage
except ImportError:
    print("Optuna is not installed. Please run: pip install optuna sqlalchemy")
    import sys
    sys.exit(1)

warnings.filterwarnings("ignore")

CFG = {
    "seed": 42,
    "num_classes": 3,
    "csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\0-data_leakage_optimized.csv",
    "mri_csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\mr\0-data_leakage_mri_clean_180.csv",
    "processed_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\data\processed_dataset_128",
    "save_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\weights",
    "n_trials": 100,
    "selection_bacc_weight": 0.6,
    "selection_acc_weight": 0.4,
    "study_name": "catboost_clinical_tuning",
    "db_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\weights\optuna_catboost.db",
}

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

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

def objective(trial: optuna.Trial, X_train, y_train, X_val, y_val, train_weights) -> float:
    param = {
        "iterations": trial.suggest_int("iterations", 200, 2000, step=100),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 30),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
    }

    model = CatBoostClassifier(
        **param,
        loss_function="MultiClass",
        classes_count=CFG["num_classes"],
        random_seed=CFG["seed"],
        verbose=0,
        task_type="GPU",
        auto_class_weights="Balanced",
    )

    train_pool = Pool(X_train, label=y_train, weight=train_weights)
    val_pool = Pool(X_val, label=y_val)

    try:
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=0)
    except Exception as e:
        print(f"Trial fail: {e}")
        return 0.0

    val_preds = model.predict(X_val).flatten().astype(int)
    acc = accuracy_score(y_val, val_preds) * 100.0
    bacc = balanced_accuracy_score(y_val, val_preds) * 100.0
    composite = (CFG["selection_bacc_weight"] * bacc) + (CFG["selection_acc_weight"] * acc)

    trial.report(composite, 0)
    return composite

def main() -> None:
    seed_everything(CFG["seed"])
    os.makedirs(CFG["save_dir"], exist_ok=True)

    # Load data once
    data_df = pd.read_csv(CFG["csv_path"])
    clinical_cols = [
        col for col in data_df.columns
        if col not in ['OASISID', 'memory', 'path'] and not str(col).startswith('Unnamed:')
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

    print(f"Train: {len(y_train)} samples | Val: {len(y_val)} samples | Features: {len(clinical_cols)}")

    storage = RDBStorage(
        url=f"sqlite:///{CFG['db_path']}",
        engine_kwargs={"connect_args": {"timeout": 30}},
    )
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=CFG["study_name"],
        load_if_exists=True,
    )

    remaining = max(0, CFG["n_trials"] - len(study.trials))
    if remaining == 0:
        print(f"All {CFG['n_trials']} trials already completed!")
    else:
        print(f"\n🚀 Starting {remaining} trials (target: {CFG['n_trials']})...")
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val, train_weights),
            n_trials=remaining,
        )

    print("\nBest Trial:")
    print(f"  Value (Composite Score): {study.best_value:.2f}%")
    print("  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    best_params_path = os.path.join(CFG["save_dir"], "7-optuna_catboost_best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\nBest parameters saved to {best_params_path}")

if __name__ == "__main__":
    main()
