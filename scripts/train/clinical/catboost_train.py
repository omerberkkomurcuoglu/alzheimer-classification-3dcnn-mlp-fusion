"""
7.1 - CatBoost Clinical Data Classification (Train)
Trains a CatBoost model on the optimized tabular clinical data using Optuna best params.
"""

import os
import random
import time
import warnings
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, auc, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import label_binarize
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

CFG = {
    "seed": 42,
    "num_classes": 3,
    "class_names": ["Normal", "Mild", "Moderate/Severe"],
    "csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\0-data_leakage_optimized.csv",
    "mri_csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\mr\0-data_leakage_mri_clean_180.csv",
    "processed_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\data\processed_dataset_128",
    "weights_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\weights",
    "model_name": "CatBoost_Clinical_Optuna",
    "early_stopping_rounds": 50,

    # Optuna best params
    "iterations": 1000,
    "depth": 4,
    "learning_rate": 0.2685324173993989,
    "l2_leaf_reg": 1.9118022269208863,
    "bagging_temperature": 4.663690617300036,
    "random_strength": 2.3766341797469517,
    "border_count": 33,
    "min_data_in_leaf": 2,
    "grow_policy": "SymmetricTree",
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

def count_labels(labels: np.ndarray) -> np.ndarray:
    return np.bincount(labels, minlength=CFG["num_classes"])

def print_split_stats(name: str, labels: np.ndarray) -> None:
    counts = count_labels(labels)
    total = int(counts.sum())
    parts = [f"{cn}={int(c)} ({100.0*c/max(total,1):.1f}%)" for cn, c in zip(CFG["class_names"], counts)]
    print(f"{name}: {len(labels)} samples | " + " | ".join(parts))

def _compute_sample_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=CFG["num_classes"])
    total = len(labels)
    class_weights = total / (CFG["num_classes"] * np.maximum(counts, 1.0))
    return class_weights[labels]

def plot_roc_curve(labels, probs, title, filename):
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
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(title); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()

def main() -> None:
    seed_everything(CFG["seed"])
    os.makedirs(CFG["weights_dir"], exist_ok=True)

    print("=" * 60)
    print(" CATBOOST CLINICAL TABULAR TRAINING ")
    print("=" * 60)

    # 1. Load & auto-update params from Optuna
    params_path = os.path.join(CFG["weights_dir"], "7-optuna_catboost_best_params.json")
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            best_params = json.load(f)
        for k, v in best_params.items():
            if k in CFG:
                CFG[k] = v
        print(f"✅ Optuna best params loaded from {params_path}")
    else:
        print("⚠️ Optuna params not found, using defaults. Run 7.0 first!")

    data_df = pd.read_csv(CFG["csv_path"])
    clinical_cols = [
        col for col in data_df.columns
        if col not in ['OASISID', 'memory', 'path'] and not str(col).startswith('Unnamed:')
    ]
    print(f"Total rows: {len(data_df)} | Clinical Features: {len(clinical_cols)}")

    # 2. Split
    train_df, val_df, test_df = make_group_splits(data_df)
    for col in clinical_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        val_df[col] = pd.to_numeric(val_df[col], errors='coerce').fillna(0)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)

    X_train, y_train = train_df[clinical_cols].values, encode_labels(train_df)
    X_val, y_val = val_df[clinical_cols].values, encode_labels(val_df)
    X_test, y_test = test_df[clinical_cols].values, encode_labels(test_df)

    print_split_stats("Train split", y_train)
    print_split_stats("Val split", y_val)
    print_split_stats("Test split", y_test)

    train_weights = _compute_sample_weights(y_train)

    # 3. Train
    print("\nTraining CatBoost Model...")
    start_time = time.time()

    model = CatBoostClassifier(
        iterations=CFG["iterations"],
        depth=CFG["depth"],
        learning_rate=CFG["learning_rate"],
        l2_leaf_reg=CFG["l2_leaf_reg"],
        bagging_temperature=CFG["bagging_temperature"],
        random_strength=CFG["random_strength"],
        border_count=CFG["border_count"],
        min_data_in_leaf=CFG["min_data_in_leaf"],
        grow_policy=CFG["grow_policy"],
        loss_function="MultiClass",
        classes_count=CFG["num_classes"],
        random_seed=CFG["seed"],
        verbose=50,
        task_type="GPU",
        auto_class_weights="Balanced",
        early_stopping_rounds=CFG["early_stopping_rounds"],
    )

    train_pool = Pool(X_train, label=y_train, weight=train_weights)
    val_pool = Pool(X_val, label=y_val)

    model.fit(train_pool, eval_set=val_pool)

    elapsed = (time.time() - start_time) / 60.0
    print(f"\nTraining completed in {elapsed:.2f} minutes.")
    print(f"Best Iteration: {model.get_best_iteration()}")

    # Save model
    best_model_path = os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_model.cbm")
    model.save_model(best_model_path)
    print(f"Model saved to {best_model_path}")

    # 4. Evaluate on Test Set
    print("\n" + "=" * 60)
    print("TEST RESULTS (CATBOOST)")
    print("=" * 60)

    test_probs = model.predict_proba(X_test)
    test_preds = model.predict(X_test).flatten().astype(int)

    acc = accuracy_score(y_test, test_preds) * 100.0
    bacc = balanced_accuracy_score(y_test, test_preds) * 100.0
    macro_f1 = f1_score(y_test, test_preds, average="macro") * 100.0

    print(f"Accuracy: {acc:.2f}% | Balanced Accuracy: {bacc:.2f}% | Macro-F1: {macro_f1:.2f}%")
    print(classification_report(y_test, test_preds, target_names=CFG["class_names"], zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=CFG["class_names"], yticklabels=CFG["class_names"])
    plt.title(f"{CFG['model_name']} - Confusion Matrix\nAccuracy: {acc:.2f}% | BAcc: {bacc:.2f}%")
    plt.ylabel('True'); plt.xlabel('Predicted'); plt.tight_layout()
    plt.savefig(os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_confusion_matrix.png"), dpi=150)
    plt.close()

    # ROC Curve
    plot_roc_curve(y_test, test_probs,
                   f"{CFG['model_name']} - Test ROC Curves",
                   os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_roc_curve.png"))

    # 5. Feature Importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (CATBOOST NATIVE)")
    print("=" * 60)

    importances = model.get_feature_importance()
    df_importance = pd.DataFrame({
        'Feature': clinical_cols, 'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    csv_imp_path = os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_feature_importance.csv")
    df_importance.to_csv(csv_imp_path, index=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=df_importance.head(20), palette='viridis')
    plt.title('Top 20 Most Important Features (CatBoost)')
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_feature_importance.png"), dpi=150)
    plt.close()
    print(f"Saved feature importance data and plot to {CFG['weights_dir']}")

if __name__ == "__main__":
    main()
