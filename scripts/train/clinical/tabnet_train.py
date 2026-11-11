"""
13.4 - TabNet Clinical Data Classification
Trains Google's TabNet architecture via the pytorch_tabnet package.
Provides interpretable attention masks and strong non-linear performance.

Note: Requires `pip install pytorch-tabnet`
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
from sklearn.preprocessing import QuantileTransformer, label_binarize
import torch

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    from pytorch_tabnet.callbacks import Callback
except ImportError:
    print("CRITICAL ERROR: 'pytorch-tabnet' is not installed.")
    print("Please run: pip install pytorch-tabnet")
    import sys
    sys.exit(1)

warnings.filterwarnings("ignore")

class CompositeScoreCallback(Callback):
    def __init__(self, weights_dir, model_name):
        super().__init__()
        self.weights_dir = weights_dir
        self.model_name = model_name
        self.best_composite = -1.0
        
    def on_epoch_end(self, epoch, logs=None):
        # 'val_balanced_accuracy' and 'val_accuracy' are keys in logs
        if 'val_accuracy' in logs and 'val_balanced_accuracy' in logs:
            val_acc = logs['val_accuracy'] * 100.0
            val_bacc = logs['val_balanced_accuracy'] * 100.0
            
            composite = (0.6 * val_bacc) + (0.4 * val_acc)
            
            if composite > self.best_composite:
                self.best_composite = composite
                # Save the best model
                path = os.path.join(self.weights_dir, f"{self.model_name}_best_composite")
                # TabNet's load_model/save_model handles .zip extension
                self.trainer.save_model(path)
                print(f"\n -> Ep {epoch:03d}: NEW BEST Composite: {composite:.2f}% (BAcc: {val_bacc:.2f}%) - Model Saved")


CFG = {
    "seed": 42,
    "num_classes": 3,
    "class_names": ["Normal", "Mild", "Moderate/Severe"],
    "csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\0-data_leakage_optimized.csv",
    "weights_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\weights",
    "model_name": "TabNet_Clinical_Optuna",
    
    # TabNet Hyperparameters (Optuna Optimized)
    "n_d": 64,
    "n_a": 64,
    "n_steps": 6,
    "gamma": 1.1823991509355174,
    "n_independent": 2,
    "n_shared": 2,
    "lambda_sparse": 0.0028875255176581136,
    "optimizer_lr": 0.003557443099162337,
    "scheduler_step": 20,
    "scheduler_gamma": 0.7,
    "patience": 15,
    "max_epochs": 100,
    "batch_size": 256,
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

def _compute_class_weights(labels: np.ndarray) -> list:
    counts = np.bincount(labels, minlength=CFG["num_classes"])
    total = len(labels)
    # PyTorch TabNet expects a list or dict of weights if sent to custom loss,
    # but TabNet handles imbalanced data automatically if we pass weights to the DataLoaders (which it builds internally)
    # Actually wait: TabNet uses standard weighted sampler internally if we provide 'weights=1'. Let's do that.
    pass

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
    print(" TABNET CLINICAL TABULAR TRAINING ")
    print("=" * 60)

    # 1. Load Data
    data_df = pd.read_csv(CFG["csv_path"])
    clinical_cols = [
        col for col in data_df.columns 
        if col not in ['OASISID', 'memory', 'path'] and not str(col).startswith('Unnamed:')
    ]
    print(f"Total rows: {len(data_df)} | Clinical Features: {len(clinical_cols)}")

    # 2. Split Data (Exactly matching)
    train_df, val_df, test_df = make_group_splits(data_df)
    
    # TabNet loves normalized data! Let's use QuantileTransformer like MLP.
    for col in clinical_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        val_df[col] = pd.to_numeric(val_df[col], errors='coerce').fillna(0)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
        
    scaler = QuantileTransformer(output_distribution='normal', random_state=CFG["seed"])
    train_df[clinical_cols] = scaler.fit_transform(train_df[clinical_cols])
    val_df[clinical_cols] = scaler.transform(val_df[clinical_cols])
    test_df[clinical_cols] = scaler.transform(test_df[clinical_cols])

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

    # 4. Initialize and Train TabNet
    print("\nTraining TabNet Model...")
    start_time = time.time()
    
    model = TabNetClassifier(
        n_d=CFG["n_d"], n_a=CFG["n_a"], n_steps=CFG["n_steps"],
        gamma=CFG["gamma"], n_independent=CFG["n_independent"], n_shared=CFG["n_shared"],
        lambda_sparse=CFG["lambda_sparse"],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=CFG["optimizer_lr"], weight_decay=1e-4),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": CFG["scheduler_step"], "gamma": CFG["scheduler_gamma"]},
        mask_type='entmax',
        device_name='cuda' if torch.cuda.is_available() else 'cpu'
    )

    composite_cb = CompositeScoreCallback(CFG["weights_dir"], CFG["model_name"])

    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=['val'],
        eval_metric=['accuracy', 'balanced_accuracy'],
        max_epochs=CFG["max_epochs"],
        patience=CFG["patience"],
        batch_size=CFG["batch_size"],
        virtual_batch_size=128,
        num_workers=0,
        weights=1, # Uses automated weighting for balancing
        drop_last=False,
        callbacks=[composite_cb]
    )
    
    elapsed = (time.time() - start_time) / 60.0
    print(f"\nTraining completed in {elapsed:.2f} minutes.")

    # Save Model
    best_model_path = os.path.join(CFG["weights_dir"], f"{CFG['model_name']}")
    saved_filepath = model.save_model(best_model_path)
    print(f"Model saved to {saved_filepath}")

    # 5. Evaluate on Test Set
    print("\n" + "="*60)
    print("TEST RESULTS (TABNET)")
    print("="*60)
    
    # Predict
    test_probs = model.predict_proba(X_test)
    test_preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, test_preds) * 100.0
    bacc = balanced_accuracy_score(y_test, test_preds) * 100.0
    macro_f1 = f1_score(y_test, test_preds, average="macro") * 100.0
    
    print(f"Accuracy: {acc:.2f}% | Balanced Accuracy: {bacc:.2f}% | Macro-F1: {macro_f1:.2f}%")
    print(classification_report(y_test, test_preds, target_names=CFG["class_names"], zero_division=0))

    # Confusion matrix with counts and percentages
    cm = confusion_matrix(y_test, test_preds)
    cm_norm = confusion_matrix(y_test, test_preds, normalize='true')
    
    # Calculate class-specific accuracies
    class_accs = cm.diagonal() / cm.sum(axis=1)
    normal_acc = class_accs[0] * 100.0
    
    plt.figure(figsize=(8, 6))
    labels = [f"{count}\n({pct:.1%})" for count, pct in zip(cm.flatten(), cm_norm.flatten())]
    labels = np.asarray(labels).reshape(3, 3)
    
    sns.heatmap(cm_norm, annot=labels, fmt="", cmap='Reds',
                xticklabels=CFG["class_names"],
                yticklabels=CFG["class_names"])
    
    plt.title(f"{CFG['model_name']} - Test Confusion Matrix\n"
              f"Accuracy: {acc:.2f}% | BAcc: {bacc:.2f}%")
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
    print("FEATURE IMPORTANCE (TABNET NATIVE)")
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
    plt.title('Top 20 Most Important Features (TabNet)')
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_feature_importance.png"), dpi=150)
    plt.close()
    print(f"Saved feature importance data and plot to {CFG['weights_dir']}")

if __name__ == "__main__":
    main()
