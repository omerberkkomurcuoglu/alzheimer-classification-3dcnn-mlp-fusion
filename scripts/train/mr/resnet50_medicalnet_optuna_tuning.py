"""
2.2 - Optuna Hyperparameter Tuning for MedicalNet ResNet50 (MRI)

Optimizes 3-phase fine-tuning pipeline to maximize Balanced Accuracy.
Trial 0 is seeded with the proven CFG from the repository (55% BAcc).
Subsequent trials explore variations around that baseline.

Custom pruning rules:
  1. Mode collapse: if only 1 class predicted after phase2_epochs+10 → prune
  2. Min BAcc: if BAcc < 40% at epoch 50 → prune
  3. Overfitting: if val_loss increases for 10 consecutive epochs after peak → prune
"""

from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from monai.transforms import (
    Compose,
    RandAffine,
    RandFlip,
    RandGaussianNoise,
    RandScaleIntensity,
)

from mri_pipeline import (
    coerce_memory_label,
    filter_existing_mri_rows,
    load_preprocessed_mri_volume,
    sanitize_augmented_mri_volume,
)

try:
    import optuna
    from optuna.storages import RDBStorage
except ImportError:
    print("Optuna is not installed. Please run: pip install optuna")
    sys.exit(1)

# ============================================================
# FIXED CONFIG (not tuned)
# ============================================================
FIXED_CFG = {
    "seed": 42,
    "num_classes": 3,
    "class_names": ["Normal", "Mild", "Moderate/Severe"],
    "processed_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\data\processed_dataset_128",
    "csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\mr\0-data_leakage_mri_clean_180.csv",
    "weights_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\mr\weights\optuna_resnet50",
    "checkpoint_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\mr\weights\optuna_resnet50\checkpoints",
    "batch_size": 4,
    "val_batch_size": 4,
    "grad_accum_steps": 4,
    "num_workers": 4,
    "epochs_per_trial": 75,
    "warmup_epochs": 3,
    "use_weighted_sampler": True,
    "use_ema": True,
    "selection_metric": "bacc",
    "n_trials": 20,
    "study_name": "resnet50_medicalnet_optuna_v1",
    "db_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\mr\weights\optuna_resnet50\optuna_resnet50.db",
}

# ============================================================
# SEED TRIAL (Trial 0) — proven CFG from repository
# ============================================================
SEED_PARAMS = {
    "head_lr": 1e-4,
    "layer4_lr": 2e-5,
    "backbone_lr": 8e-6,
    "phase2_head_lr": 5e-5,
    "phase3_head_lr": 8e-5,
    "dropout": 0.30,
    "label_smoothing": 0.05,
    "sampler_power": 0.75,
    "weight_decay": 3e-4,
    "phase1_epochs": 10,
    "phase2_epochs": 30,
    "ema_decay": 0.998,
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
# AUGMENTATION
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
    return np.bincount(labels, minlength=FIXED_CFG["num_classes"])


def print_split_stats(name: str, df: pd.DataFrame) -> None:
    counts = count_mapped_labels(df)
    total = int(counts.sum())
    parts = []
    for class_name, count in zip(FIXED_CFG["class_names"], counts):
        pct = 100.0 * count / max(total, 1)
        parts.append(f"{class_name}={int(count)} ({pct:.1f}%)")
    print(f"{name}: {len(df)} samples | " + " | ".join(parts))


def make_group_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = np.array([encode_memory_label(v) for v in df["memory"]], dtype=np.int64)
    groups = df["OASISID"].astype(str).to_numpy()
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=FIXED_CFG["seed"])
        train_val_idx, test_idx = next(outer.split(df, labels, groups))
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        inner_labels = np.array(
            [encode_memory_label(v) for v in train_val_df["memory"]], dtype=np.int64
        )
        inner_groups = train_val_df["OASISID"].astype(str).to_numpy()
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=FIXED_CFG["seed"])
        train_idx, val_idx = next(inner.split(train_val_df, inner_labels, inner_groups))
        return (
            train_val_df.iloc[train_idx].reset_index(drop=True),
            train_val_df.iloc[val_idx].reset_index(drop=True),
            test_df,
        )
    except Exception:
        outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=FIXED_CFG["seed"])
        train_val_idx, test_idx = next(outer.split(df, groups=groups))
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        inner = GroupShuffleSplit(n_splits=1, test_size=15 / 85, random_state=FIXED_CFG["seed"])
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

        backbone = monai_r50(
            spatial_dims=3, n_input_channels=1, num_classes=1,
            shortcut_type="B", bias_downsample=False,
        )
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
def build_param_groups(
    model: BrainClassifier, layer3_lr: float, layer4_lr: float, head_lr: float, weight_decay: float
) -> list[dict]:
    groups = []
    for name, lr in (("layer3", layer3_lr), ("layer4", layer4_lr), ("head", head_lr)):
        params = [
            param
            for param_name, param in model.named_parameters()
            if param.requires_grad and name in param_name
        ]
        if params:
            groups.append({"params": params, "lr": lr, "weight_decay": weight_decay})
    return groups


def set_warmup_lrs(optimizer: optim.Optimizer, epoch_in_phase: int, target_lrs: list[float]) -> None:
    scale = min(1.0, float(epoch_in_phase + 1) / float(max(FIXED_CFG["warmup_epochs"], 1)))
    for group, target_lr in zip(optimizer.param_groups, target_lrs):
        group["lr"] = target_lr * scale


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
        "pred_counts": np.bincount(
            np.asarray(preds_all, dtype=np.int64), minlength=FIXED_CFG["num_classes"]
        ).tolist(),
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
            loss = criterion(outputs, labels) / FIXED_CFG["grad_accum_steps"]

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = (step + 1) % FIXED_CFG["grad_accum_steps"] == 0 or (step + 1) == len(loader)
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
        total_loss += loss.item() * FIXED_CFG["grad_accum_steps"] * batch_size
        total_count += batch_size
        total_correct += (outputs.argmax(1) == labels).sum().item()
        loop.set_postfix(
            train_loss=f"{total_loss / max(total_count, 1):.4f}",
            train_acc=f"{100.0 * total_correct / max(total_count, 1):.2f}%",
        )
    return total_loss / max(total_count, 1), 100.0 * total_correct / max(total_count, 1)


# ============================================================
# CHECKPOINT SYSTEM
# ============================================================
def save_checkpoint(
    trial_number: int,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler | None,
    ema: EMAModel | None,
    best_score: float,
    phase_start_epoch: int,
) -> str:
    os.makedirs(FIXED_CFG["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(FIXED_CFG["checkpoint_dir"], f"checkpoint_trial_{trial_number}.pth")
    
    # EMA backup parameters for resuming
    ema_shadow = {k: v.cpu().clone() for k, v in ema.shadow.items()} if ema else None
    
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler else None,
        "ema_shadow": ema_shadow,
        "best_score": best_score,
        "phase_start_epoch": phase_start_epoch,
    }, ckpt_path)
    return ckpt_path


def load_checkpoint(
    trial_number: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler | None,
    ema: EMAModel | None,
) -> dict | None:
    ckpt_path = os.path.join(FIXED_CFG["checkpoint_dir"], f"checkpoint_trial_{trial_number}.pth")
    if not os.path.exists(ckpt_path):
        return None
    
    print(f"  🔄 Checkpoint bulundu: {ckpt_path}. Kaldığı yerden yükleniyor...")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler and checkpoint["scaler_state"]:
        scaler.load_state_dict(checkpoint["scaler_state"])
    if ema and checkpoint["ema_shadow"]:
        ema.shadow = {k: v.to(DEVICE) for k, v in checkpoint["ema_shadow"].items()}
        
    return checkpoint


def remove_checkpoint(trial_number: int):
    ckpt_path = os.path.join(FIXED_CFG["checkpoint_dir"], f"checkpoint_trial_{trial_number}.pth")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


# ============================================================
# CUSTOM PRUNER — mode collapse, min BAcc, overfitting
# ============================================================
class SmartResNetPruner:
    """
    Custom pruning logic based on lessons learned from previous training runs.
    Called after each epoch to decide whether to prune the trial.
    """

    def __init__(self, phase2_epochs: int):
        self.phase2_epochs = phase2_epochs
        self.best_val_loss = float("inf")
        self.val_loss_increase_streak = 0

    def should_prune(self, epoch: int, val_metrics: dict) -> tuple[bool, str]:
        """Returns (should_prune, reason)."""
        bacc = val_metrics["bacc"]
        pred_counts = val_metrics["pred_counts"]
        val_loss = val_metrics["loss"]

        # --- Kural 1: Mode Collapse Kontrolü ---
        # Artık phase2_epochs'a değil, sabit Epoch 45'e (index 44) bakıyoruz
        if epoch >= 44:
            num_predicted_classes = sum(1 for c in pred_counts if c > 0)
            if num_predicted_classes <= 1:
                return True, f"MODE COLLAPSE: Epoch {epoch+1}'de hala tek sınıf tahmin ediliyor"

        # 60. epoktan önce başka sebeple kesme (Kullanıcı isteği)
        if epoch < 59:
            return False, ""

        # --- Kural 2: Minimum BAcc Kontrolü (60+ epok) ---
        if bacc < 40.0:
            return True, f"MIN_BACC: Epoch {epoch+1}'de BAcc={bacc:.1f}% < %40 minimum eşiği"

        # --- Kural 3: Overfitting Kontrolü (60+ epok) ---
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.val_loss_increase_streak = 0
        else:
            self.val_loss_increase_streak += 1

        if self.val_loss_increase_streak >= 15:
            return True, (
                f"OVERFITTING: Val Loss {self.val_loss_increase_streak} epok boyunca artıyor "
                f"(best={self.best_val_loss:.4f}, current={val_loss:.4f})"
            )

        return False, ""


# ============================================================
# OBJECTIVE FUNCTION
# ============================================================
def create_data_loaders():
    """Create data loaders once and reuse across trials."""
    raw_df = pd.read_csv(FIXED_CFG["csv_path"])
    df = filter_existing_mri_rows(FIXED_CFG["processed_dir"], raw_df)
    dropped_missing = len(raw_df) - len(df)
    if dropped_missing:
        print(f"Dropped {dropped_missing} rows with missing MRI tensors before splitting.")

    train_df, val_df, test_df = make_group_splits(df)
    print_split_stats("Train split", train_df)
    print_split_stats("Val split", val_df)
    print_split_stats("Test split", test_df)

    train_ds = BrainMRIDataset(FIXED_CFG["processed_dir"], train_df, augment=True)
    val_ds = BrainMRIDataset(FIXED_CFG["processed_dir"], val_df, augment=False)
    test_ds = BrainMRIDataset(FIXED_CFG["processed_dir"], test_df, augment=False)

    train_counts = count_mapped_labels(train_ds.df)
    print(f"Train class counts: {dict(zip(FIXED_CFG['class_names'], train_counts.tolist()))}")

    return train_ds, val_ds, test_ds, train_counts


def build_loaders(train_ds, val_ds, train_counts, sampler_power):
    """Build loaders with a specific sampler_power."""
    loader_kwargs = {
        "num_workers": FIXED_CFG["num_workers"],
        "pin_memory": True,
        "persistent_workers": FIXED_CFG["num_workers"] > 0,
    }

    if FIXED_CFG["use_weighted_sampler"]:
        targets = np.array(
            [encode_memory_label(v) for v in train_ds.df["memory"]], dtype=np.int64
        )
        weights = 1.0 / np.power(train_counts.astype(np.float64) + 1.0, sampler_power)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights[targets], dtype=torch.double), len(targets), replacement=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=FIXED_CFG["batch_size"], sampler=sampler, drop_last=False, **loader_kwargs
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=FIXED_CFG["batch_size"], shuffle=True, drop_last=False, **loader_kwargs
        )

    val_loader = DataLoader(
        val_ds, batch_size=FIXED_CFG["val_batch_size"], shuffle=False, **loader_kwargs
    )

    return train_loader, val_loader


def objective(trial: optuna.Trial, train_ds, val_ds, test_ds, train_counts) -> float:
    """Single trial: train ResNet50 with sampled hyperparameters."""
    trial_start = time.time()

    # --- Sample hyperparameters ---
    hp = {
        "head_lr": trial.suggest_float("head_lr", 5e-5, 2e-4, log=True),
        "layer4_lr": trial.suggest_float("layer4_lr", 5e-6, 5e-5, log=True),
        "backbone_lr": trial.suggest_float("backbone_lr", 2e-6, 2e-5, log=True), # Phase3'te Layer3 için LR
        "phase2_head_lr": trial.suggest_float("phase2_head_lr", 2e-5, 1e-4, log=True),
        "phase3_head_lr": trial.suggest_float("phase3_head_lr", 3e-5, 1.5e-4, log=True),
        "dropout": trial.suggest_float("dropout", 0.45, 0.60), # ARTIRILDI: Ezberlemeyi önlemek için ama collapse yapmayacak seviyede
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15),
        "sampler_power": trial.suggest_float("sampler_power", 0.60, 0.90),
        "weight_decay": trial.suggest_float("weight_decay", 5e-4, 5e-3, log=True), # ARTIRILDI: Ağırlık büyümesini engellemek için
        "phase1_epochs": trial.suggest_int("phase1_epochs", 5, 15, step=5),
        "phase2_epochs": trial.suggest_int("phase2_epochs", 20, 35, step=5),
        "ema_decay": trial.suggest_float("ema_decay", 0.995, 0.999),
    }

    print(f"\n{'='*70}")
    print(f"TRIAL {trial.number} — Hiperparametreler:")
    for k, v in hp.items():
        print(f"  {k}: {v}")
    print(f"{'='*70}")

    # --- Build data loaders with trial's sampler_power ---
    train_loader, val_loader = build_loaders(train_ds, val_ds, train_counts, hp["sampler_power"])

    # --- Build model ---
    model = BrainClassifier(FIXED_CFG["num_classes"], hp["dropout"], pretrained=True).to(DEVICE)
    model.freeze_batch_norms()
    criterion = nn.CrossEntropyLoss(label_smoothing=hp["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
    ema = EMAModel(model, hp["ema_decay"]) if FIXED_CFG["use_ema"] else None

    best_score = 0.0
    best_state_dict = None
    pruner = SmartResNetPruner(hp["phase2_epochs"])

    optimizer = None
    scheduler = None
    target_lrs: list[float] = []
    phase_start_epoch = 0

    total_epochs = FIXED_CFG["epochs_per_trial"]
    start_epoch = 0
    phase_start_epoch = 0

    # Resuming logic için ilk setup (PH1 defaultları ile)
    model.freeze_backbone()
    model.freeze_batch_norms()
    optimizer = optim.AdamW(model.head.parameters(), lr=hp["head_lr"], weight_decay=hp["weight_decay"])
    target_lrs = [hp["head_lr"]]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, hp["phase1_epochs"] - FIXED_CFG["warmup_epochs"]), eta_min=hp["head_lr"] * 0.1)

    checkpoint = load_checkpoint(trial.number, model, optimizer, scheduler, scaler, ema)
    if checkpoint:
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint.get("best_score", 0.0)
        phase_start_epoch = checkpoint["phase_start_epoch"]
        print(f"  🚀 Trial {trial.number} Epoch {start_epoch+1}'den devam ediyor...")

    for epoch in range(start_epoch, total_epochs):
        # --- Phase transitions ---
        if epoch == 0:
            pass # Setup zaten yapıldı

        elif epoch == hp["phase1_epochs"]:
            phase_start_epoch = epoch
            model.unfreeze_from("layer4")
            model.freeze_batch_norms()
            if FIXED_CFG["use_ema"]: ema = EMAModel(model, hp["ema_decay"])
            optimizer = optim.AdamW(build_param_groups(model, 0.0, hp["layer4_lr"], hp["phase2_head_lr"], hp["weight_decay"]))
            target_lrs = [group["lr"] for group in optimizer.param_groups]
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, hp["phase2_epochs"] - hp["phase1_epochs"] - FIXED_CFG["warmup_epochs"]), eta_min=min(target_lrs) * 0.1)

        elif epoch == hp["phase2_epochs"]:
            phase_start_epoch = epoch
            # LAYER3'Ü DE ÇÖZÜYORUZ
            model.unfreeze_from("layer3")
            model.freeze_batch_norms()
            if FIXED_CFG["use_ema"]: ema = EMAModel(model, hp["ema_decay"])
            optimizer = optim.AdamW(build_param_groups(model, hp["backbone_lr"], hp["layer4_lr"], hp["phase3_head_lr"], hp["weight_decay"]))
            target_lrs = [group["lr"] for group in optimizer.param_groups]
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - hp["phase2_epochs"] - FIXED_CFG["warmup_epochs"]), eta_min=min(target_lrs) * 0.1)

        assert optimizer is not None and scheduler is not None
        phase_epoch = epoch - phase_start_epoch
        if phase_epoch < FIXED_CFG["warmup_epochs"]:
            set_warmup_lrs(optimizer, phase_epoch, target_lrs)

        # --- Train one epoch ---
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, ema)

        # --- Evaluate ---
        if ema is not None:
            ema.apply_shadow(model)
            val_metrics = evaluate(model, val_loader, criterion)
            ema.restore(model)
        else:
            val_metrics = evaluate(model, val_loader, criterion)

        if phase_epoch >= FIXED_CFG["warmup_epochs"]:
            scheduler.step()

        # --- Track best ---
        current_score = 0.70 * val_metrics["bacc"] + 0.30 * val_metrics["acc"]
        
        if current_score > best_score:
            best_score = float(current_score)
            if ema is not None: ema.apply_shadow(model)
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if ema is not None: ema.restore(model)

        # --- Log ---
        tag = "PH1" if epoch < hp["phase1_epochs"] else ("PH2" if epoch < hp["phase2_epochs"] else "PH3")
        lr_text = " ".join(f"{g['lr']:.2e}" for g in optimizer.param_groups)
        elapsed = (time.time() - start_time) / 60.0
        pc = val_metrics["pred_counts"]
        print(f"  T{trial.number} [{tag}] Ep {epoch+1:03d} | TrL {train_loss:.4f} TrA {train_acc:5.2f}% | VaL {val_metrics['loss']:.4f} VaA {val_metrics['acc']:5.2f}% BAcc {val_metrics['bacc']:5.2f}% | LR [{lr_text}] | Pred N:{pc[0]} Mi:{pc[1]} MS:{pc[2]} | {elapsed:.1f}m | Comp={current_score:.2f}% (best={best_score:.2f}%)")

        # --- Report to Optuna ---
        trial.report(best_score, epoch)

        # --- Checkpoint ---
        save_checkpoint(trial.number, epoch, model, optimizer, scheduler, scaler, ema, best_score, phase_start_epoch)

        # --- Pruning ---
        should_prune, reason = pruner.should_prune(epoch, val_metrics)
        if should_prune:
            print(f"  ⛔ PRUNED: {reason}")
            _save_trial_weights(trial.number, best_state_dict, best_score, hp)
            raise optuna.exceptions.TrialPruned(reason)

        if trial.should_prune():
            print(f"  ⛔ PRUNED by Optuna MedianPruner at epoch {epoch+1}")
            _save_trial_weights(trial.number, best_state_dict, best_score, hp)
            raise optuna.exceptions.TrialPruned()

    # --- Trial tamamlandı ---
    trial_elapsed = (time.time() - trial_start) / 60.0

    # --- Test seti değerlendirmesi ---
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state_dict.items()})
    test_loader = DataLoader(
        test_ds, batch_size=FIXED_CFG["val_batch_size"], shuffle=False,
        num_workers=FIXED_CFG["num_workers"], pin_memory=True,
    )
    test_metrics = evaluate(model, test_loader, criterion)
    test_composite = 0.70 * test_metrics["bacc"] + 0.30 * test_metrics["acc"]
    val_composite = best_score  # Eğitim sırasındaki en iyi val composite

    # TEST SIZINTISINI ENGELLEMEK İÇİN SADECE VAL COMPOSITE DÖNDÜR!
    final_score = val_composite
    val_test_gap = val_composite - test_composite

    pc_test = test_metrics["pred_counts"]
    print(f"\n  ✅ Trial {trial.number} tamamlandı | Süre: {trial_elapsed:.1f}m")
    print(f"     Val  Composite : {val_composite:.2f}% (Optuna Hedefi)")
    print(f"     Test Composite : {test_composite:.2f}% (Acc={test_metrics['acc']:.2f}% BAcc={test_metrics['bacc']:.2f}%) [SADECE BİLGİ AMAÇLI]")
    print(f"     Test Preds     : N:{pc_test[0]} Mi:{pc_test[1]} MS:{pc_test[2]}")
    print(f"     Val→Test Gap   : {val_test_gap:+.2f}%")

    # En iyi ağırlıkları kaydet
    _save_trial_weights(trial.number, best_state_dict, best_score, hp, test_metrics=test_metrics)

    # Store extra metrics
    trial.set_user_attr("val_composite", val_composite)
    trial.set_user_attr("test_acc", test_metrics["acc"])
    trial.set_user_attr("test_bacc", test_metrics["bacc"])
    trial.set_user_attr("test_composite", test_composite)
    trial.set_user_attr("val_test_gap", val_test_gap)
    trial.set_user_attr("final_score", final_score)
    trial.set_user_attr("trial_duration_min", trial_elapsed)

    # GPU belleğini temizle
    del model, ema, optimizer, scheduler, scaler, criterion, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    return final_score

def _save_trial_weights(
    trial_number: int,
    state_dict: dict | None,
    best_score: float,
    hp: dict,
    test_metrics: dict | None = None,
) -> None:
    """Save trial's best weights and metadata (+ test results if available)."""
    os.makedirs(FIXED_CFG["weights_dir"], exist_ok=True)

    if state_dict is not None:
        weight_path = os.path.join(
            FIXED_CFG["weights_dir"], f"trial_{trial_number:02d}_best.pth"
        )
        torch.save(state_dict, weight_path)

    meta_path = os.path.join(
        FIXED_CFG["weights_dir"], f"trial_{trial_number:02d}_meta.json"
    )
    meta = {
        "trial_number": trial_number,
        "best_score": best_score,
        "hyperparameters": hp,
    }
    if test_metrics is not None:
        test_composite = 0.70 * test_metrics["bacc"] + 0.30 * test_metrics["acc"]
        meta["test_acc"] = round(test_metrics["acc"], 4)
        meta["test_bacc"] = round(test_metrics["bacc"], 4)
        meta["test_composite"] = round(test_composite, 4)
        meta["final_score"] = round(0.40 * best_score + 0.60 * test_composite, 4)
        meta["val_test_gap"] = round(best_score - test_composite, 4)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    seed_everything(FIXED_CFG["seed"])
    print(f"Device: {DEVICE}")
    os.makedirs(FIXED_CFG["weights_dir"], exist_ok=True)

    # --- Veri setini bir kez yükle ---
    train_ds, val_ds, test_ds, train_counts = create_data_loaders()

    # --- Optuna Study oluştur (SQLite ile kalıcı) ---
    storage = RDBStorage(
        url=f"sqlite:///{FIXED_CFG['db_path']}",
        engine_kwargs={"connect_args": {"timeout": 30}},
    )

    study = optuna.create_study(
        study_name=FIXED_CFG["study_name"],
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,      # İlk 3 trial pruning yok (seed dahil)
            n_warmup_steps=60,       # İlk 60 epoch pruning yok (40 -> 60)
            interval_steps=5,        # Her 5 epoch'ta kontrol et
        ),
    )

    # --- FAIL/Interrupted Trial Recovery (Restart from Scratch) ---
    failed_trials = [t for t in study.trials if t.state in (optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.FAIL)]
    for ft in failed_trials:
        ckpt_path = os.path.join(FIXED_CFG["checkpoint_dir"], f"checkpoint_trial_{ft.number}.pth")
        if os.path.exists(ckpt_path): os.remove(ckpt_path)
        print(f"  ↩️ Trial {ft.number} sıfırlanıyor ve en baştan başlatılmak üzere sıraya alınıyor...")
        study.enqueue_trial(ft.params)

    # --- Trial 0: Seed ---
    if len(study.trials) == 0:
        print("\n🌱 Trial 0 seed ediliyor...")
        study.enqueue_trial(SEED_PARAMS)

    # --- Optuna çalıştır ---
    remaining_trials = max(0, FIXED_CFG["n_trials"] - len(study.trials))
    if remaining_trials == 0:
        print(f"\nTüm {FIXED_CFG['n_trials']} trial zaten tamamlanmış!")
    else:
        print(f"\n🚀 {remaining_trials} trial başlatılıyor (toplam hedef: {FIXED_CFG['n_trials']})...")
        study.optimize(
            lambda trial: objective(trial, train_ds, val_ds, test_ds, train_counts),
            n_trials=remaining_trials,
            gc_after_trial=True,
        )

    # --- Sonuçları özetle ---
    print("\n" + "=" * 70)
    print("OPTUNA SONUÇ ÖZETİ")
    print("=" * 70)

    print(f"\nToplam trial sayısı: {len(study.trials)}")
    print(f"Tamamlanan trial sayısı: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Budanan (pruned) trial sayısı: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

    # En iyi 5 trial
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        completed_trials.sort(key=lambda t: t.value or 0, reverse=True)
        print("\n--- En İyi 5 Trial (Final = 0.4×Val + 0.6×Test) ---")
        for i, t in enumerate(completed_trials[:5]):
            duration = t.user_attrs.get("trial_duration_min", "?")
            val_c = t.user_attrs.get("val_composite", None)
            test_c = t.user_attrs.get("test_composite", None)
            test_b = t.user_attrs.get("test_bacc", None)
            gap = t.user_attrs.get("val_test_gap", None)
            val_s = f"{val_c:.2f}" if val_c is not None else "N/A"
            test_s = f"{test_c:.2f}" if test_c is not None else "N/A"
            bacc_s = f"{test_b:.2f}" if test_b is not None else "N/A"
            gap_s = f"{gap:+.2f}" if gap is not None else "N/A"
            print(f"  #{i+1} Trial {t.number}: Final={t.value:.2f}% | Val={val_s}% Test={test_s}% (BAcc={bacc_s}%) Gap={gap_s}% | {duration}m")
            for k, v in t.params.items():
                print(f"       {k}: {v}")

        # En iyi trial'ın ağırlıklarını best_overall olarak kopyala
        best_trial = completed_trials[0]
        src = os.path.join(FIXED_CFG["weights_dir"], f"trial_{best_trial.number:02d}_best.pth")
        dst = os.path.join(FIXED_CFG["weights_dir"], "best_overall.pth")
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, dst)
            print(f"\n🏆 En iyi ağırlıklar kopyalandı: {dst}")

        # JSON olarak kaydet
        best_params_path = os.path.join(FIXED_CFG["weights_dir"], "best_params.json")
        with open(best_params_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "trial_number": best_trial.number,
                    "best_score": best_trial.value,
                    "params": best_trial.params,
                },
                f,
                indent=4,
                ensure_ascii=False,
            )
        print(f"📋 En iyi parametreler kaydedildi: {best_params_path}")
    else:
        print("\n⚠️ Hiçbir trial tamamlanamadı!")


if __name__ == "__main__":
    main()
