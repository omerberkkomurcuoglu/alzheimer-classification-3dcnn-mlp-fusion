"""
13.3 - Random Forest Clinical Data Classification
Trains a RandomForestClassifier on the optimized tabular clinical data.
Extremely robust to noise and great for an ensemble alongside MLP and XGBoost.
"""

import os
import random
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import label_binarize
import joblib

warnings.filterwarnings("ignore")

CFG = {
    "seed": 42,
    "num_classes": 3,
    "class_names": ["Normal", "Mild", "Moderate/Severe"],
    "csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\0-data_leakage_optimized.csv",
    "weights_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\weights",
    "model_name": "RandomForest_Clinical_Optuna",
    
    # Random Forest Hyperparameters (Optuna Optimized)
    "n_estimators": 800,
    "max_depth": 18,
    "min_samples_split": 12,
    "min_samples_leaf": 9,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "criterion": "entropy",
    "n_jobs": -1,
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
        try:
            v_f = float(v)
        except Exception:
            v_f = 0.0
        labels.append(label_map.get(v_f, 0))
    return np.array(labels, dtype=np.int64)

def count_labels(labels: np.ndarray) -> np.ndarray:
    return np.bincount(labels, minlength=CFG["num_classes"])

def print_split_stats(name: str, labels: np.ndarray) -> None:
    counts = count_labels(labels)
    total = int(counts.sum())
    parts = []
    for class_name, count in zip(CFG["class_names"], counts):
        pct = 100.0 * count / max(total, 1)
        parts.append(f"{class_name}={int(count)} ({pct:.1f}%)")
    print(f"{name}: {len(labels)} samples | " + " | ".join(parts))

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

def main() -> None:
    seed_everything(CFG["seed"])
    os.makedirs(CFG["weights_dir"], exist_ok=True)

    print("=" * 60)
    print(" RANDOM FOREST CLINICAL TABULAR TRAINING ")
    print("=" * 60)

    # 1. Load Data
    data_df = pd.read_csv(CFG["csv_path"])
    clinical_cols = [
        col for col in data_df.columns 
        if col not in ['OASISID', 'memory', 'path'] and not str(col).startswith('Unnamed:')
    ]
    print(f"Total rows: {len(data_df)} | Clinical Features: {len(clinical_cols)}")

    # 2. Split Data (Exactly matching MLP and XGBoost)
    train_df, val_df, test_df = make_group_splits(data_df)
    
    # 3. Extract Features and Labels
    X_train = train_df[clinical_cols].values
    y_train = encode_labels(train_df)
    
    X_val = val_df[clinical_cols].values
    y_val = encode_labels(val_df)
    
    X_test = test_df[clinical_cols].values
    y_test = encode_labels(test_df)

    print_split_stats("Train split", y_train)
    print_split_stats("Val split", y_val)
    print_split_stats("Test split", y_test)

    # 4. Initialize and Train Random Forest Model
    print("\nTraining Random Forest Model (using all available CPU cores)...")
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=CFG["n_estimators"],
        max_depth=CFG["max_depth"],
        min_samples_split=CFG["min_samples_split"],
        min_samples_leaf=CFG["min_samples_leaf"],
        max_features=CFG["max_features"],
        class_weight=CFG["class_weight"],
        criterion=CFG["criterion"],
        n_jobs=CFG["n_jobs"],
        random_state=CFG["seed"],
    )

    # Note: Random Forest doesn't 'early stop' based on validation set natively like Boosters.
    # It trains independent trees. We can merge Train + Val here to give it more data optimally,
    # but to keep the 1:1 fairness with XGBoost, we'll just train on X_train.
    model.fit(X_train, y_train)
    
    elapsed = (time.time() - start_time) / 60.0
    print(f"\nTraining completed in {elapsed:.2f} minutes.")

    # Save Model
    best_model_path = os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_model.pkl")
    joblib.dump(model, best_model_path)
    print(f"Model saved to {best_model_path}")

    # 5. Evaluate on Test Set
    print("\n" + "="*60)
    print("TEST RESULTS (RANDOM FOREST)")
    print("="*60)
    
    # Predict
    test_probs = model.predict_proba(X_test)
    test_preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, test_preds) * 100.0
    bacc = balanced_accuracy_score(y_test, test_preds) * 100.0
    macro_f1 = f1_score(y_test, test_preds, average="macro") * 100.0
    
    print(f"Accuracy: {acc:.2f}% | Balanced Accuracy: {bacc:.2f}% | Macro-F1: {macro_f1:.2f}%")
    print(classification_report(y_test, test_preds, target_names=CFG["class_names"], zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=CFG["class_names"],
                yticklabels=CFG["class_names"])
    plt.title(f"{CFG['model_name']} - Confusion Matrix\nAccuracy: {acc:.2f}% | BAcc: {bacc:.2f}%")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_confusion_matrix.png"), dpi=150)
    plt.close()
    
    # ROC Curve
    roc_title = f"{CFG['model_name']} - Test ROC Curves"
    roc_path = os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_roc_curve.png")
    plot_roc_curve(y_test, test_probs, roc_title, roc_path)

    # 6. Feature Importance Extract
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (RANDOM FOREST NATIVE)")
    print("="*60)
    
    importances = model.feature_importances_
    df_importance = pd.DataFrame({
        'Feature': clinical_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    csv_imp_path = os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_feature_importance.csv")
    df_importance.to_csv(csv_imp_path, index=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=df_importance.head(20), palette='viridis')
    plt.title('Top 20 Most Important Features (Random Forest)')
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_feature_importance.png"), dpi=150)
    plt.close()
    print(f"Saved feature importance data and plot to {CFG['weights_dir']}")

if __name__ == "__main__":
    main()
