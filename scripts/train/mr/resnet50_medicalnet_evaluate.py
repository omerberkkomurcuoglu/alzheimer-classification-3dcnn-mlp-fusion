"""
14.4.1 - Evaluate the stable MedicalNet ResNet50 checkpoint on the 14.4 splits.

This script mirrors the 14.4 training data filtering, group split logic, and
model architecture, then loads an already-trained checkpoint for validation and
test evaluation only.
"""

from __future__ import annotations

import os
import sys
import random
import warnings

# Add parent directory to sys.path to find mri_pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mri_pipeline import coerce_memory_label, filter_existing_mri_rows, load_preprocessed_mri_volume


CFG = {
    "seed": 42,
    "num_classes": 3,
    "class_names": ["Normal", "Mild", "Moderate/Severe"],
    "processed_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\data\processed_dataset_128",
    "csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\mr\0-data_leakage_mri_clean_180.csv",
    "weights_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\mr\weights",
    "model_name": "ResNet50_MedicalNet",
    "dropout": 0.30,
    "batch_size": 4,
    "num_workers": 0,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BrainMRIDataset(Dataset):
    def __init__(self, root_dir: str, df: pd.DataFrame) -> None:
        self.root_dir = root_dir
        self.df = filter_existing_mri_rows(root_dir, df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        npy_path = os.path.join(self.root_dir, f"{row['path']}.npy")
        volume = load_preprocessed_mri_volume(npy_path)
        label = coerce_memory_label(row["memory"])
        return torch.from_numpy(volume).unsqueeze(0), torch.tensor(label, dtype=torch.long)


def encode_memory_label(value) -> int:
    return coerce_memory_label(value)


def count_mapped_labels(df: pd.DataFrame) -> np.ndarray:
    labels = np.array([encode_memory_label(v) for v in df["memory"]], dtype=np.int64)
    return np.bincount(labels, minlength=CFG["num_classes"])


def print_split_stats(name: str, df: pd.DataFrame) -> None:
    counts = count_mapped_labels(df)
    total = int(counts.sum())
    parts = []
    for class_name, count in zip(CFG["class_names"], counts):
        pct = 100.0 * count / max(total, 1)
        parts.append(f"{class_name}={int(count)} ({pct:.1f}%)")
    print(f"{name}: {len(df)} samples | " + " | ".join(parts))


def make_group_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = np.array([encode_memory_label(v) for v in df["memory"]], dtype=np.int64)
    groups = df["OASISID"].astype(str).to_numpy()
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG["seed"])
        train_val_idx, test_idx = next(outer.split(df, labels, groups))
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        inner_labels = np.array([encode_memory_label(v) for v in train_val_df["memory"]], dtype=np.int64)
        inner_groups = train_val_df["OASISID"].astype(str).to_numpy()
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=CFG["seed"])
        train_idx, val_idx = next(inner.split(train_val_df, inner_labels, inner_groups))
        return (
            train_val_df.iloc[train_idx].reset_index(drop=True),
            train_val_df.iloc[val_idx].reset_index(drop=True),
            test_df,
        )
    except Exception:
        outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=CFG["seed"])
        train_val_idx, test_idx = next(outer.split(df, groups=groups))
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        inner = GroupShuffleSplit(n_splits=1, test_size=15 / 85, random_state=CFG["seed"])
        train_idx, val_idx = next(inner.split(train_val_df, groups=train_val_df["OASISID"]))
        return (
            train_val_df.iloc[train_idx].reset_index(drop=True),
            train_val_df.iloc[val_idx].reset_index(drop=True),
            test_df,
        )


class BrainClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        from monai.networks.nets import resnet50 as monai_r50

        backbone = monai_r50(spatial_dims=3, n_input_channels=1, num_classes=1, shortcut_type="B", bias_downsample=False)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act = backbone.act
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.act(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.head(x)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, split_name: str) -> dict[str, object]:
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    probs: list[np.ndarray] = []

    for images, batch_labels in tqdm(loader, desc=split_name):
        images = images.to(DEVICE, non_blocking=True)
        outputs = model(images)
        batch_probs = torch.softmax(outputs, dim=1)
        preds.extend(batch_probs.argmax(1).cpu().numpy().tolist())
        labels.extend(batch_labels.numpy().tolist())
        probs.extend(batch_probs.cpu().numpy())

    label_arr = np.array(labels)
    pred_arr = np.array(preds)
    prob_arr = np.array(probs)
    return {
        "acc": 100.0 * accuracy_score(label_arr, pred_arr),
        "bacc": 100.0 * balanced_accuracy_score(label_arr, pred_arr),
        "macro_f1": 100.0 * f1_score(label_arr, pred_arr, average="macro", zero_division=0),
        "labels": label_arr,
        "preds": pred_arr,
        "probs": prob_arr,
    }


def pick_checkpoint_path() -> str:
    candidates = [
        os.path.join(CFG["weights_dir"], "resnet.pth"),
        os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_best_balanced.pth"),
        os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_best_composite.pth"),
        os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_best_acc.pth"),
        os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_best_loss.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No checkpoint found for model '{CFG['model_name']}' in {CFG['weights_dir']}")


def save_confusion_matrix(metrics: dict[str, object], split_name: str) -> str:
    cm = confusion_matrix(metrics["labels"], metrics["preds"])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CFG["class_names"], yticklabels=CFG["class_names"],
                annot_kws={"size": 13, "weight": "bold"},
                cbar_kws={'label': 'Number of Samples'})
    
    plt.title(f"ResNet50 MedicalNet - {split_name}\nAccuracy: {metrics['acc']:.2f}% | Balanced Acc: {metrics['bacc']:.2f}%", 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=25, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    out_path = os.path.abspath(f"{CFG['model_name']}_{split_name.lower()}_confusion_matrix.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def save_roc_curve(metrics: dict[str, object], split_name: str) -> str:
    y_true = metrics["labels"]
    y_score = metrics["probs"]
    
    y_true_bin = label_binarize(y_true, classes=range(CFG["num_classes"]))
    n_classes = y_true_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    
    # Plot classes with specific colors from reference
    colors = ['#2196F3', '#FF9800', '#E91E63'] # Blue, Orange, Pink/Magenta
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{CFG["class_names"][i]} (AUC = {roc_auc[i]:.3f})')

    # Plot micro-average as black dotted line
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-avg (AUC = {roc_auc["micro"]:.3f})',
             color='black', linestyle=':', linewidth=3)

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curve: {split_name}\nF1: {metrics["macro_f1"]:.1f}% | BAcc: {metrics["bacc"]:.1f}%', 
              fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)
    plt.grid(True, linestyle='-', alpha=0.2)
    
    plt.tight_layout()
    out_path = os.path.abspath(f"{CFG['model_name']}_{split_name.lower()}_roc_curve.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def report_metrics(split_name: str, metrics: dict[str, object]) -> None:
    print(f"\n{split_name} RESULTS")
    print(
        f"Accuracy: {metrics['acc']:.2f}% | "
        f"Balanced Accuracy: {metrics['bacc']:.2f}% | "
        f"Macro-F1: {metrics['macro_f1']:.2f}%"
    )
    print(
        classification_report(
            metrics["labels"],
            metrics["preds"],
            target_names=CFG["class_names"],
            zero_division=0,
        )
    )
    cm_path = save_confusion_matrix(metrics, split_name)
    roc_path = save_roc_curve(metrics, split_name)
    print(f"Saved confusion matrix: {cm_path}")
    print(f"Saved ROC curve: {roc_path}")


def main() -> None:
    seed_everything(CFG["seed"])
    print(f"Device: {DEVICE}")

    raw_df = pd.read_csv(CFG["csv_path"])
    df = filter_existing_mri_rows(CFG["processed_dir"], raw_df)
    dropped_missing = len(raw_df) - len(df)
    if dropped_missing:
        print(f"Dropped {dropped_missing} rows with missing MRI tensors before splitting.")
    duplicate_paths = int(df["path"].duplicated().sum())
    if duplicate_paths:
        raise ValueError(f"Clean CSV still has duplicate MRI paths: {duplicate_paths}")

    train_df, val_df, test_df = make_group_splits(df)
    print_split_stats("Train split", train_df)
    print_split_stats("Val split", val_df)
    print_split_stats("Test split", test_df)

    val_ds = BrainMRIDataset(CFG["processed_dir"], val_df)
    test_ds = BrainMRIDataset(CFG["processed_dir"], test_df)

    loader_kwargs = {"num_workers": CFG["num_workers"], "pin_memory": DEVICE.type == "cuda"}
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False, **loader_kwargs)

    model = BrainClassifier(CFG["num_classes"], CFG["dropout"]).to(DEVICE)
    checkpoint_path = pick_checkpoint_path()
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    print(f"Loaded checkpoint: {checkpoint_path}")

    val_metrics = evaluate(model, val_loader, "VAL")
    test_metrics = evaluate(model, test_loader, "TEST")

    report_metrics("VAL", val_metrics)
    report_metrics("TEST", test_metrics)


if __name__ == "__main__":
    main()
