"""
1.1 - Custom 3D CNN — MRI Image Classification
Class-Weighted CrossEntropyLoss variant (no Focal Loss).
"""

from __future__ import annotations

import os
import random
import sys
import time
import warnings
from pathlib import Path

# Add the parent directory to sys.path to allow importing mri_pipeline
sys.path.append(str(Path(__file__).resolve().parent.parent))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from monai.transforms import Compose, RandAffine, RandFlip, RandGaussianNoise, RandScaleIntensity

from mri_pipeline import (
    coerce_memory_label,
    filter_existing_mri_rows,
    load_preprocessed_mri_volume,
    sanitize_augmented_mri_volume,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from custom_model import Custom3DCNN

CFG = {
    "seed": 42,
    "num_classes": 3,
    "class_names": ["Normal", "Mild", "Moderate/Severe"],
    "processed_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\data\processed_dataset_128",
    "csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\mr\0-data_leakage_mri_clean_180.csv",
    "weights_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\mr\weights",
    "model_name": "Custom3DCNN_v3_optim_sampler075",
    "batch_size": 24,
    "val_batch_size": 24,
    "test_batch_size": 24,
    "grad_accum_steps": 2, 
    "num_workers": 2,
    "epochs": 150,
    "patience": 35,
    "min_epochs_before_early_stop": 40,
    "dropout": 0.25,
    "label_smoothing": 0.10,
    "lr": 1e-4,
    "weight_decay": 3e-5,
    "use_weighted_sampler": True,
    "sampler_power": 0.75,
    "use_ema": True,
    "ema_decay": 0.998,
    "selection_metric": "bacc",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class EMAModel:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.data.clone()
                continue
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

def build_ce_criterion(class_weights: torch.Tensor, label_smoothing: float) -> nn.CrossEntropyLoss:
    """Build CrossEntropyLoss with class weights and label smoothing."""
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

def build_train_transforms() -> Compose:
    return Compose([
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.2, spatial_axis=1),
        RandAffine(prob=0.20, rotate_range=(0.04, 0.04, 0.04), scale_range=(0.04, 0.04, 0.04), mode="bilinear", padding_mode="zeros"),
        RandGaussianNoise(prob=0.10, mean=0.0, std=0.01),
        RandScaleIntensity(factors=0.08, prob=0.10)
    ])

class BrainMRIDataset(Dataset):
    def __init__(self, root_dir: str, df: pd.DataFrame, augment: bool = False) -> None:
        self.root_dir = root_dir
        self.df = filter_existing_mri_rows(root_dir, df)
        self.augment = augment
        self.transforms = build_train_transforms() if augment else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        npy_path = os.path.join(self.root_dir, f"{row['path']}.npy")
        volume = load_preprocessed_mri_volume(npy_path)
            
        label = coerce_memory_label(row["memory"])
        if self.augment and self.transforms is not None:
            volume = volume[np.newaxis, ...]
            volume = self.transforms(volume)
            if not isinstance(volume, np.ndarray):
                volume = volume.numpy()
            volume = sanitize_augmented_mri_volume(volume.squeeze(0))
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

def build_class_alpha(counts: np.ndarray) -> torch.Tensor:
    safe = np.maximum(counts.astype(np.float32), 1.0)
    weights = safe.sum() / safe
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

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
        return train_val_df.iloc[train_idx].reset_index(drop=True), train_val_df.iloc[val_idx].reset_index(drop=True), test_df
    except Exception:
        outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=CFG["seed"])
        train_val_idx, test_idx = next(outer.split(df, groups=groups))
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        inner = GroupShuffleSplit(n_splits=1, test_size=15 / 85, random_state=CFG["seed"])
        train_idx, val_idx = next(inner.split(train_val_df, groups=train_val_df["OASISID"]))
        return train_val_df.iloc[train_idx].reset_index(drop=True), train_val_df.iloc[val_idx].reset_index(drop=True), test_df

class CustomCNNClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        # Import the architecture dynamically like the original script
        self.backbone = Custom3DCNN(num_classes=num_classes, dropout=dropout)
        self.freeze_all_bns = False

    def freeze_batch_norms(self) -> None:
        self.freeze_all_bns = True
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                if module.weight is not None:
                    module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False

    def train(self, mode: bool = True) -> "CustomCNNClassifier":
        super().train(mode)
        if mode and self.freeze_all_bns:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

def compute_selection_score(metrics: dict[str, float]) -> float:
    return float(metrics["bacc"])

def predict_with_tta(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """TTA: original + 3 axis flips -> average softmax"""
    predictions = []
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            predictions.append(F.softmax(model(images), dim=1))
            predictions.append(F.softmax(model(torch.flip(images, [2])), dim=1))
            predictions.append(F.softmax(model(torch.flip(images, [3])), dim=1))
            predictions.append(F.softmax(model(torch.flip(images, [4])), dim=1))
    return torch.stack(predictions).mean(dim=0)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, use_tta: bool = False) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    preds_all, labels_all = [], []
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            if use_tta:
                probs = predict_with_tta(model, images)
                outputs = torch.log(probs + 1e-8) # Convert back to log-probs for loss estimation
            else:
                outputs = model(images)
            loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)
        preds_all.extend(outputs.argmax(1).cpu().numpy().tolist())
        labels_all.extend(labels.cpu().numpy().tolist())
    return {
        "loss": total_loss / max(total_count, 1),
        "acc": 100.0 * accuracy_score(labels_all, preds_all),
        "bacc": 100.0 * balanced_accuracy_score(labels_all, preds_all),
        "macro_f1": 100.0 * f1_score(labels_all, preds_all, average="macro", zero_division=0),
        "preds": preds_all,
        "labels": labels_all,
        "pred_counts": np.bincount(np.asarray(preds_all, dtype=np.int64), minlength=CFG["num_classes"]).tolist(),
    }

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    epoch_idx: int,
    ema: EMAModel | None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    optimizer.zero_grad(set_to_none=True)
    loop = tqdm(loader, desc=f"Ep {epoch_idx + 1}", leave=False)
    for step, (images, labels) in enumerate(loop):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, labels) / CFG["grad_accum_steps"]
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = (step + 1) % CFG["grad_accum_steps"] == 0 or (step + 1) == len(loader)
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)

        batch_size = labels.size(0)
        total_loss += loss.item() * CFG["grad_accum_steps"] * batch_size
        total_count += batch_size
        total_correct += (outputs.argmax(1) == labels).sum().item()
        loop.set_postfix(
            train_loss=f"{total_loss / max(total_count, 1):.4f}",
            train_acc=f"{100.0 * total_correct / max(total_count, 1):.2f}%",
        )
    return total_loss / max(total_count, 1), 100.0 * total_correct / max(total_count, 1)

def main() -> None:
    seed_everything(CFG["seed"])
    print(f"Device: {DEVICE}")

    raw_df = pd.read_csv(CFG["csv_path"])
    df = filter_existing_mri_rows(CFG["processed_dir"], raw_df)
    
    train_df, val_df, test_df = make_group_splits(df)
    print_split_stats("Train split", train_df)
    print_split_stats("Val split", val_df)
    print_split_stats("Test split", test_df)

    train_ds = BrainMRIDataset(CFG["processed_dir"], train_df, augment=True)
    val_ds = BrainMRIDataset(CFG["processed_dir"], val_df, augment=False)
    test_ds = BrainMRIDataset(CFG["processed_dir"], test_df, augment=False)

    train_counts = count_mapped_labels(train_ds.df)
    alpha = build_class_alpha(train_counts)

    loader_kwargs = {"num_workers": CFG["num_workers"], "pin_memory": True, "persistent_workers": CFG["num_workers"] > 0}
    if CFG["use_weighted_sampler"]:
        targets = np.array([encode_memory_label(v) for v in train_ds.df["memory"]], dtype=np.int64)
        weights = 1.0 / np.power(train_counts.astype(np.float64) + 1.0, CFG["sampler_power"])
        sampler = WeightedRandomSampler(torch.as_tensor(weights[targets], dtype=torch.double), len(targets), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler, drop_last=False, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=CFG["val_batch_size"], shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=CFG["test_batch_size"], shuffle=False, **loader_kwargs)

    model = CustomCNNClassifier(CFG["num_classes"], CFG["dropout"]).to(DEVICE)
    # Çift ağırlıklandırmayı kaldırdık, sadece smooth var! (Sampler açık olduğu için ağırlığa gerek yok)
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min=1e-6)

    best = {"score": -float("inf"), "bacc": -float("inf"), "acc": -float("inf"), "loss": float("inf")}
    patience_left = CFG["patience"]
    ema = EMAModel(model, CFG["ema_decay"]) if CFG["use_ema"] else None

    for epoch in range(CFG["epochs"]):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, ema)
        if ema is not None:
            ema.apply_shadow(model)
            val_metrics = evaluate(model, val_loader, criterion)
            ema.restore(model)
        else:
            val_metrics = evaluate(model, val_loader, criterion)
        
        scheduler.step()

        selection_score = compute_selection_score(val_metrics)
        improved = False
        os.makedirs(CFG["weights_dir"], exist_ok=True)
        
        if selection_score > best["score"]:
            if ema is not None:
                ema.apply_shadow(model)
            best["score"] = selection_score
            improved = True
            if ema is not None:
                ema.restore(model)
        if val_metrics["bacc"] > best["bacc"]:
            if ema is not None:
                ema.apply_shadow(model)
            best["bacc"] = float(val_metrics["bacc"])
            torch.save(model.state_dict(), os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_best_balanced.pth"))
            improved = True
            if ema is not None:
                ema.restore(model)
        if val_metrics["acc"] > best["acc"]:
            best["acc"] = float(val_metrics["acc"])
        if val_metrics["loss"] < best["loss"]:
            best["loss"] = float(val_metrics["loss"])
        
        patience_left = CFG["patience"] if improved else patience_left - 1

        lr_text = f"{optimizer.param_groups[0]['lr']:.2e}"
        elapsed = (time.time() - start_time) / 60.0
        pc = val_metrics["pred_counts"]
        print(
            f"Ep {epoch + 1:03d} | CE(w) | "
            f"TrL {train_loss:.4f} TrA {train_acc:5.2f}% | "
            f"VaL {val_metrics['loss']:.4f} VaA {val_metrics['acc']:5.2f}% "
            f"BAcc {val_metrics['bacc']:5.2f}% MF1 {val_metrics['macro_f1']:5.2f}% | "
            f"LR [{lr_text}] | Pred N:{pc[0]} Mi:{pc[1]} MS:{pc[2]} | "
            f"{elapsed:.1f}m | pat={patience_left}"
        )

        if epoch + 1 >= CFG["min_epochs_before_early_stop"] and patience_left <= 0:
            print(f"Early stopping triggered on {CFG['selection_metric']}")
            break

    best_path = os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_best_balanced.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    
    # Run Normal Evaluation
    test_metrics = evaluate(model, test_loader, criterion, use_tta=False)
    # Run TTA Evaluation
    tta_metrics = evaluate(model, test_loader, criterion, use_tta=True)

    print("\nTEST RESULTS (NORMAL)")
    print(
        f"Accuracy: {test_metrics['acc']:.2f}% | Balanced Accuracy: {test_metrics['bacc']:.2f}% | "
        f"Macro-F1: {test_metrics['macro_f1']:.2f}%"
    )
    print(classification_report(test_metrics["labels"], test_metrics["preds"], target_names=CFG["class_names"], zero_division=0))

    print("\nTEST RESULTS (TTA - TEST TIME AUGMENTATION)")
    print(
        f"Accuracy: {tta_metrics['acc']:.2f}% | Balanced Accuracy: {tta_metrics['bacc']:.2f}% | "
        f"Macro-F1: {tta_metrics['macro_f1']:.2f}%"
    )
    print(classification_report(tta_metrics["labels"], tta_metrics["preds"], target_names=CFG["class_names"], zero_division=0))

    use_tta = tta_metrics['bacc'] > test_metrics['bacc']
    best_cm_metrics = tta_metrics if use_tta else test_metrics
    tag = "TTA" if use_tta else "Normal"

    cm = confusion_matrix(best_cm_metrics["labels"], best_cm_metrics["preds"])
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{CFG['model_name']} ({tag})")
    plt.colorbar()
    plt.xticks(range(CFG["num_classes"]), CFG["class_names"], rotation=20)
    plt.yticks(range(CFG["num_classes"]), CFG["class_names"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_confusion_matrix.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
