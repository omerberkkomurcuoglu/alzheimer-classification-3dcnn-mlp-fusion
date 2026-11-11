"""
14.4 - Stable MedicalNet ResNet50 training on cleaned MRI-level CSV.

This is a copy-track experiment that leaves 14.3 untouched.
Main changes:
- uses the clean one-row-per-MRI CSV
- freezes all BatchNorm layers for tiny 3D batches
- updates EMA per optimizer step
- uses stratified group split when available
- keeps augmentation and loss milder at the start
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


CFG = {
    "seed": 42,
    "num_classes": 3,
    "class_names": ["Normal", "Mild", "Moderate/Severe"],
    "processed_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\data\processed_dataset_128",
    "csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\mr\0-data_leakage_mri_clean_180.csv",
    "weights_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\mr\weights",
    "model_name": "ResNet50_MedicalNet_v9_ce_unweighted",
    "batch_size": 4,
    "val_batch_size": 4,
    "test_batch_size": 4,
    "grad_accum_steps": 4,
    "num_workers": 4,
    "epochs": 120,
    "patience": 30,
    "min_epochs_before_early_stop": 35,
    "dropout": 0.30,
    "label_smoothing": 0.05,
    "backbone_lr": 8e-6,
    "layer4_lr": 2e-5,
    "head_lr": 1e-4,
    "phase2_head_lr": 5e-5,
    "phase3_head_lr": 8e-5,
    "phase1_epochs": 10,
    "phase2_epochs": 30,
    "warmup_epochs": 3,
    "weight_decay": 3e-4,
    "use_weighted_sampler": True,
    "sampler_power": 0.75,
    "use_ema": True,
    "ema_decay": 0.998,
    "selection_metric": "bacc",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# SEED
# ============================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================
# EMA
# ============================================================
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


# ============================================================
# AUGMENTATION (v10 style)
# ============================================================
def build_train_transforms() -> Compose:
    return Compose(
        [
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.2, spatial_axis=1),
            RandAffine(
                prob=0.20,
                rotate_range=(0.04, 0.04, 0.04),
                scale_range=(0.04, 0.04, 0.04),
                mode="bilinear",
                padding_mode="zeros",
            ),
            RandGaussianNoise(prob=0.10, mean=0.0, std=0.01),
            RandScaleIntensity(factors=0.08, prob=0.10),
        ]
    )


# ============================================================
# DATASET
# ============================================================
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


# ============================================================
# HELPERS
# ============================================================
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


# ============================================================
# MODEL (MedicalNet ResNet50)
# ============================================================
def download_medicalnet_r50() -> dict[str, torch.Tensor]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="TencentMedicalNet/MedicalNet-Resnet50",
        filename="resnet_50_23dataset.pth",
    )
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint if "state_dict" not in checkpoint else checkpoint["state_dict"]
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


class BrainClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float, pretrained: bool = True) -> None:
        super().__init__()
        from monai.networks.nets import resnet50 as monai_r50

        backbone = monai_r50(spatial_dims=3, n_input_channels=1, num_classes=1, shortcut_type="B", bias_downsample=False)
        if pretrained:
            mednet = download_medicalnet_r50()
            own = backbone.state_dict()
            for name, tensor in mednet.items():
                if name in own and own[name].shape == tensor.shape:
                    own[name] = tensor
            backbone.load_state_dict(own)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act = backbone.act
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.freeze_all_bns = False
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

    def freeze_backbone(self) -> None:
        for layer in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_from(self, stage: str) -> None:
        order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
        start = order.index(stage)
        for name in order[:start]:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
        for name in order[start:]:
            for param in getattr(self, name).parameters():
                param.requires_grad = True
        if self.freeze_all_bns:
            self.freeze_batch_norms()

    def freeze_batch_norms(self) -> None:
        self.freeze_all_bns = True
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                if module.weight is not None:
                    module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False

    def train(self, mode: bool = True) -> "BrainClassifier":
        super().train(mode)
        if mode and self.freeze_all_bns:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.act(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.head(x)


# ============================================================
# TRAINING UTILITIES
# ============================================================
def build_param_groups(model: BrainClassifier, layer3_lr: float, layer4_lr: float, head_lr: float) -> list[dict]:
    groups = []
    for name, lr in (("layer3", layer3_lr), ("layer4", layer4_lr), ("head", head_lr)):
        params = [param for param_name, param in model.named_parameters() if param.requires_grad and name in param_name]
        if params:
            groups.append({"params": params, "lr": lr, "weight_decay": CFG["weight_decay"]})
    return groups


def set_warmup_lrs(optimizer: optim.Optimizer, epoch_in_phase: int, target_lrs: list[float]) -> None:
    scale = min(1.0, float(epoch_in_phase + 1) / float(max(CFG["warmup_epochs"], 1)))
    for group, target_lr in zip(optimizer.param_groups, target_lrs):
        group["lr"] = target_lr * scale


def compute_selection_score(metrics: dict[str, float]) -> float:
    if CFG["selection_metric"] == "bacc":
        return float(metrics["bacc"])
    return (
        float(CFG["selection_bacc_weight"]) * float(metrics["bacc"])
        + float(CFG["selection_macro_f1_weight"]) * float(metrics["macro_f1"])
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    preds_all, labels_all = [], []
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
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


def save_if_better(model: nn.Module, ema: EMAModel | None, metrics: dict[str, object], best: dict[str, float]) -> dict[str, bool]:
    improved = {"score": False, "bacc": False, "acc": False, "loss": False}
    os.makedirs(CFG["weights_dir"], exist_ok=True)
    if ema is not None:
        ema.apply_shadow(model)
    selection_score = compute_selection_score(metrics)
    if selection_score > best["score"]:
        torch.save(model.state_dict(), os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_best_composite.pth"))
        improved["score"] = True
    if metrics["bacc"] > best["bacc"]:
        torch.save(model.state_dict(), os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_best_balanced.pth"))
        improved["bacc"] = True
    if metrics["acc"] > best["acc"]:
        torch.save(model.state_dict(), os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_best_acc.pth"))
        improved["acc"] = True
    if metrics["loss"] < best["loss"]:
        torch.save(model.state_dict(), os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_best_loss.pth"))
        improved["loss"] = True
    if ema is not None:
        ema.restore(model)
    return improved


# ============================================================
# MAIN — 3-Phase Training
# ============================================================
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

    train_ds = BrainMRIDataset(CFG["processed_dir"], train_df, augment=True)
    val_ds = BrainMRIDataset(CFG["processed_dir"], val_df, augment=False)
    test_ds = BrainMRIDataset(CFG["processed_dir"], test_df, augment=False)

    train_counts = count_mapped_labels(train_ds.df)
    print(f"Train class counts: {dict(zip(CFG['class_names'], train_counts.tolist()))}")

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

    model = BrainClassifier(CFG["num_classes"], CFG["dropout"], pretrained=True).to(DEVICE)
    model.freeze_batch_norms()
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
    ema = EMAModel(model, CFG["ema_decay"]) if CFG["use_ema"] else None

    best = {"score": -float("inf"), "bacc": -float("inf"), "acc": -float("inf"), "loss": float("inf")}
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_bacc": [], "val_macro_f1": [], "val_score": []}
    patience_left = CFG["patience"]
    optimizer = None
    scheduler = None
    target_lrs: list[float] = []
    phase_start_epoch = 0

    for epoch in range(CFG["epochs"]):
        if epoch == 0:
            phase_start_epoch = 0
            model.freeze_backbone()
            model.freeze_batch_norms()
            optimizer = optim.AdamW(model.head.parameters(), lr=CFG["head_lr"], weight_decay=CFG["weight_decay"])
            target_lrs = [CFG["head_lr"]]
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, CFG["phase1_epochs"] - CFG["warmup_epochs"]), eta_min=CFG["head_lr"] * 0.1)
        elif epoch == CFG["phase1_epochs"]:
            phase_start_epoch = epoch
            model.unfreeze_from("layer4")
            model.freeze_batch_norms()
            optimizer = optim.AdamW(build_param_groups(model, 0.0, CFG["layer4_lr"], CFG["phase2_head_lr"]))
            target_lrs = [group["lr"] for group in optimizer.param_groups]
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, CFG["phase2_epochs"] - CFG["phase1_epochs"] - CFG["warmup_epochs"]), eta_min=min(target_lrs) * 0.1)
        elif epoch == CFG["phase2_epochs"]:
            phase_start_epoch = epoch
            model.unfreeze_from("layer3")
            model.freeze_batch_norms()
            optimizer = optim.AdamW(build_param_groups(model, CFG["backbone_lr"], CFG["layer4_lr"], CFG["phase3_head_lr"]))
            target_lrs = [group["lr"] for group in optimizer.param_groups]
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, CFG["epochs"] - CFG["phase2_epochs"] - CFG["warmup_epochs"]), eta_min=min(target_lrs) * 0.1)

        assert optimizer is not None and scheduler is not None
        phase_epoch = epoch - phase_start_epoch
        if phase_epoch < CFG["warmup_epochs"]:
            set_warmup_lrs(optimizer, phase_epoch, target_lrs)

        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, ema)
        if ema is not None:
            ema.apply_shadow(model)
            val_metrics = evaluate(model, val_loader, criterion)
            ema.restore(model)
        else:
            val_metrics = evaluate(model, val_loader, criterion)

        if phase_epoch >= CFG["warmup_epochs"]:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_metrics["acc"])
        history["val_bacc"].append(val_metrics["bacc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_score"].append(compute_selection_score(val_metrics))

        improved = save_if_better(model, ema, val_metrics, best)
        if improved["score"]:
            best["score"] = float(compute_selection_score(val_metrics))
        if improved["bacc"]:
            best["bacc"] = float(val_metrics["bacc"])
        if improved["acc"]:
            best["acc"] = float(val_metrics["acc"])
        if improved["loss"]:
            best["loss"] = float(val_metrics["loss"])
        patience_left = CFG["patience"] if improved["score"] else patience_left - 1

        tag = "PH1" if epoch < CFG["phase1_epochs"] else ("PH2" if epoch < CFG["phase2_epochs"] else "PH3")
        lr_text = " ".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)
        elapsed = (time.time() - start_time) / 60.0
        pred_counts = val_metrics["pred_counts"]
        sel_score = compute_selection_score(val_metrics)
        print(
            f"[{tag}] Ep {epoch + 1:03d} | CE(w) | "
            f"TrL {train_loss:.4f} TrA {train_acc:5.2f}% | "
            f"VaL {val_metrics['loss']:.4f} VaA {val_metrics['acc']:5.2f}% "
            f"BAcc {val_metrics['bacc']:5.2f}% MF1 {val_metrics['macro_f1']:5.2f}% Sel {sel_score:5.2f} | "
            f"LR [{lr_text}] | Pred N:{pred_counts[0]} Mi:{pred_counts[1]} MS:{pred_counts[2]} | "
            f"{elapsed:.1f}m | pat={patience_left}"
        )

        if epoch + 1 >= CFG["min_epochs_before_early_stop"] and patience_left <= 0:
            print(f"Early stopping triggered on {CFG['selection_metric']}")
            break

    best_path = os.path.join(
        CFG["weights_dir"],
        f"{CFG['model_name']}_{'best_balanced' if CFG['selection_metric'] == 'bacc' else 'best_composite'}.pth",
    )
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_metrics = evaluate(model, test_loader, criterion)
    print("\nTEST RESULTS")
    print(
        f"Accuracy: {test_metrics['acc']:.2f}% | Balanced Accuracy: {test_metrics['bacc']:.2f}% | "
        f"Macro-F1: {test_metrics['macro_f1']:.2f}% | "
        f"Selection Score: {compute_selection_score(test_metrics):.2f}"
    )
    print(classification_report(test_metrics["labels"], test_metrics["preds"], target_names=CFG["class_names"], zero_division=0))

    cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"])
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(CFG["model_name"])
    plt.colorbar()
    plt.xticks(range(CFG["num_classes"]), CFG["class_names"], rotation=20)
    plt.yticks(range(CFG["num_classes"]), CFG["class_names"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(f"{CFG['model_name']}_confusion_matrix.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
