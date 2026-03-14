"""
MedicalNet ResNet50 (3D) — Test Script
Loads best_balanced.pth and generates:
  - Classification Report
  - Confusion Matrix
  - AUC-ROC Curves
  - TTA (Test-Time Augmentation) comparison

Model architecture matches 14.3-train_resnet50.py (BrainClassifier)
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.transforms import Compose, RandFlip, RandAffine
from mri_pipeline import load_normalized_mri_volume

# ============================================================
# CONFIG (must match 14.3 training script)
# ============================================================
CFG = {
    "label_map": {0: 0, 0.5: 1, 1: 2, 2: 2},
    "class_names": ["Normal", "Mild", "Moderate/Severe"],
    "num_classes": 3,
    "processed_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\processed_dataset_128",
    "csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\12-data_leakage.csv",
    "model_name": "ResNet50_MedicalNet_v3",
    "weights_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\weights",
    "dropout": 0.4,
    "batch_size": 4,
    "tta_folds": 5,
}

# ============================================================
# 1. DATASET (same as 14.3)
# ============================================================
class BrainMRIDataset(Dataset):
    def __init__(self, root_dir: str, df: pd.DataFrame):
        self.root = root_dir
        self.label_map = CFG["label_map"]

        valid = []
        for i, row in df.iterrows():
            if os.path.exists(os.path.join(root_dir, f"{row['path']}.npy")):
                valid.append(i)
        self.df = df.loc[valid].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.root, f"{row['path']}.npy")

        vol = load_normalized_mri_volume(path)

        label_raw = float(row["memory"]) if not pd.isna(row["memory"]) else 0.0
        label = self.label_map.get(label_raw, 0)

        tensor = torch.from_numpy(vol).unsqueeze(0)  # (1, D, H, W)
        return tensor, torch.tensor(label, dtype=torch.long)

# ============================================================
# 2. MODEL (exact copy from 14.3-train_resnet50.py)
# ============================================================
def _download_medicalnet_r50():
    from huggingface_hub import hf_hub_download
    print("MedicalNet ResNet50 weights downloading …")
    path = hf_hub_download(
        repo_id="TencentMedicalNet/MedicalNet-Resnet50",
        filename="resnet_50_23dataset.pth",
    )
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt if isinstance(ckpt, dict) and "state_dict" not in ckpt else ckpt.get("state_dict", ckpt)
    clean = {k.replace("module.", ""): v for k, v in sd.items()}
    return clean


class BrainClassifier(nn.Module):
    """
    MedicalNet ResNet50 backbone -> custom head.
    (Architecture identical to 14.3-train_resnet50.py)
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.5, pretrained: bool = True):
        super().__init__()

        from monai.networks.nets import resnet50 as monai_r50
        backbone = monai_r50(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1,
            shortcut_type="B",
            bias_downsample=False,
        )

        if pretrained:
            mednet = _download_medicalnet_r50()
            own = backbone.state_dict()
            loaded = 0
            for k, v in mednet.items():
                if k in own and own[k].shape == v.shape:
                    own[k] = v
                    loaded += 1
            backbone.load_state_dict(own)
            print(f"✅ Loaded {loaded} MedicalNet layers")

        # expose backbone layers individually
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.act     = backbone.act
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4

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
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        last = self.head[-1]
        nn.init.normal_(last.weight, 0, 0.01)
        nn.init.zeros_(last.bias)

    def forward(self, x):
        x = self.maxpool(self.act(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.head(x)

# ============================================================
# 3. TTA (Test-Time Augmentation) — same as 14.3
# ============================================================
def build_tta_transforms():
    """5 lightweight test-time augmentations (including identity)."""
    return [
        None,
        Compose([RandFlip(prob=1.0, spatial_axis=0)]),
        Compose([RandFlip(prob=1.0, spatial_axis=1)]),
        Compose([RandFlip(prob=1.0, spatial_axis=2)]),
        Compose([RandAffine(
            prob=1.0,
            rotate_range=(0.05, 0.05, 0.05),
            mode="bilinear",
            padding_mode="zeros",
        )]),
    ]


@torch.no_grad()
def predict_with_tta(model, images, n_folds=5):
    """TTA: apply transforms and average softmax probabilities."""
    transforms = build_tta_transforms()[:n_folds]
    probs = torch.zeros(images.size(0), CFG["num_classes"], device=images.device)

    for tfm in transforms:
        if tfm is not None:
            batch = []
            for img in images:
                a = tfm(img.cpu().numpy())
                batch.append(torch.from_numpy(a) if isinstance(a, np.ndarray) else a)
            batch = torch.stack(batch).to(images.device)
        else:
            batch = images
        probs += F.softmax(model(batch), dim=1)

    probs /= n_folds
    return probs

# ============================================================
# 4. EVALUATION FUNCTION
# ============================================================
def evaluate_set(model, loader, device, set_name, class_names, save_prefix, use_tta=False):
    """
    Evaluate a dataset split:
    - Classification Report
    - Confusion Matrix
    - AUC-ROC Curves
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"{set_name}"):
            images = images.to(device)

            if use_tta:
                probs = predict_with_tta(model, images, CFG["tta_folds"])
            else:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    num_classes = len(class_names)

    # --- Metrics ---
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    tta_tag = " (TTA)" if use_tta else ""
    print(f"\n{'='*60}")
    print(f"{set_name}{tta_tag} RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Balanced Accuracy: {bal_acc*100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{set_name}{tta_tag} - Confusion Matrix\nAcc: {acc*100:.2f}% | Balanced Acc: {bal_acc*100:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_confusion_matrix.png", dpi=150)
    print(f"✅ Confusion Matrix saved: {save_prefix}_confusion_matrix.png")

    # --- AUC-ROC Curves ---
    labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))

    plt.figure(figsize=(8, 6))
    colors = ['#2196F3', '#FF9800', '#E91E63']

    auc_scores = {}
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[class_names[i]] = roc_auc
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(labels_bin.ravel(), all_probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, color='black', lw=2, linestyle='--',
             label=f'Micro-avg (AUC = {roc_auc_micro:.3f})')

    plt.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{set_name}{tta_tag} - ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_roc_curve.png", dpi=150)
    print(f"✅ ROC Curve saved: {save_prefix}_roc_curve.png")

    print(f"\nAUC Scores:")
    for name, score in auc_scores.items():
        print(f"  {name}: {score:.3f}")
    print(f"  Micro-avg: {roc_auc_micro:.3f}")

    return acc, bal_acc, auc_scores

# ============================================================
# 5. MAIN
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Paths
    processed_dir = CFG["processed_dir"]
    data_csv = pd.read_csv(CFG["csv_path"])
    MODEL_NAME = CFG["model_name"]

    # CNN-only: deduplicate (same as training)
    before = len(data_csv)
    data_csv = data_csv.drop_duplicates(subset="path", keep="first").reset_index(drop=True)
    print(f"Total samples: {len(data_csv)} (after dedup: removed {before - len(data_csv)})")

    # Data split (same as training: 70% Train, 15% Val, 15% Test)
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_inds, test_inds = next(splitter_test.split(data_csv, groups=data_csv['OASISID']))

    train_val_df = data_csv.iloc[train_val_inds].reset_index(drop=True)
    test_df = data_csv.iloc[test_inds].reset_index(drop=True)

    val_ratio = 15 / 85
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_inds, val_inds = next(splitter_val.split(train_val_df, groups=train_val_df['OASISID']))

    val_df = train_val_df.iloc[val_inds].reset_index(drop=True)

    # Datasets
    val_dataset = BrainMRIDataset(processed_dir, val_df)
    test_dataset = BrainMRIDataset(processed_dir, test_df)

    print(f"Validation dataset: {len(val_dataset)}")
    print(f"Test dataset: {len(test_dataset)}")

    # DataLoaders
    BATCH_SIZE = CFG["batch_size"]
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model (same architecture as training, pretrained=False since we load trained weights)
    model = BrainClassifier(
        num_classes=CFG["num_classes"],
        dropout=CFG["dropout"],
        pretrained=False,
    )
    model.to(device)

    # Load weights from weights directory
    weights_path = os.path.join(CFG["weights_dir"], f"{MODEL_NAME}_best_balanced.pth")

    if not os.path.exists(weights_path):
        print(f"\n❌ Weight file not found: {weights_path}")
        print("Please train the model first using '14.3-train_resnet50.py'")
        exit()

    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"\n✅ Model loaded successfully: {weights_path}")
    except RuntimeError as e:
        print(f"\n❌ Model loading error: {e}")
        exit()

    model.eval()
    class_names = CFG["class_names"]

    # ===== VALIDATION (Normal) =====
    print("\n🚀 Validation Evaluation...")
    val_acc, val_bal, val_aucs = evaluate_set(
        model, val_loader, device,
        set_name="VALIDATION",
        class_names=class_names,
        save_prefix=f"{MODEL_NAME}_val"
    )

    # ===== TEST (Normal) =====
    print("\n🚀 Test Evaluation (Normal)...")
    test_acc, test_bal, test_aucs = evaluate_set(
        model, test_loader, device,
        set_name="TEST",
        class_names=class_names,
        save_prefix=f"{MODEL_NAME}_test"
    )

    # ===== TEST (TTA) =====
    print("\n🚀 Test Evaluation (TTA)...")
    tta_acc, tta_bal, tta_aucs = evaluate_set(
        model, test_loader, device,
        set_name="TEST",
        class_names=class_names,
        save_prefix=f"{MODEL_NAME}_test_tta",
        use_tta=True
    )

    # ===== SUMMARY =====
    print("\n" + "="*70)
    print(f"  {MODEL_NAME} — GENERAL SUMMARY")
    print("="*70)
    print(f"{'Metric':<25} {'Validation':>12} {'Test':>12} {'Test (TTA)':>12}")
    print("-"*62)
    print(f"{'Accuracy':<25} {val_acc*100:>11.2f}% {test_acc*100:>11.2f}% {tta_acc*100:>11.2f}%")
    print(f"{'Balanced Accuracy':<25} {val_bal*100:>11.2f}% {test_bal*100:>11.2f}% {tta_bal*100:>11.2f}%")
    for name in class_names:
        v = val_aucs.get(name, 0)
        t = test_aucs.get(name, 0)
        tt = tta_aucs.get(name, 0)
        print(f"{'AUC ' + name:<25} {v:>12.3f} {t:>12.3f} {tt:>12.3f}")

    # Best result
    best_tag = "TTA" if tta_bal > test_bal else "Normal"
    best_bacc = max(tta_bal, test_bal)
    print(f"\n✅ Best Test Result ({best_tag}): Balanced Accuracy = {best_bacc*100:.2f}%")

    plt.show()
    print("\n✅ Completed!")
