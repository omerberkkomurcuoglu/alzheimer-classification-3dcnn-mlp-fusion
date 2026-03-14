"""
14.3 — Optimal 3D MRI Classification for OASIS-3 Alzheimer's Detection

Architecture : MedicalNet ResNet50 (3D, pretrained on 23 medical datasets)
Training     : 3-phase → Head warmup → Fine-tune → SWA
Augmentation : MONAI (Flip, Affine, Noise, Contrast, CoarseDropout)
Loss         : CrossEntropyLoss + label_smoothing
Evaluation   : Test-Time Augmentation (TTA, 5-fold)

Designed for maximum balanced accuracy on imbalanced 3-class problem.
"""

import os, sys, time, random, copy, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix,
    balanced_accuracy_score, accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from monai.transforms import (
    Compose, RandFlip, RandAffine, RandGaussianNoise,
    RandScaleIntensity, RandAdjustContrast, RandCoarseDropout,
)
from mri_pipeline import load_normalized_mri_volume

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
CFG = {
    "seed": 42,
    "num_classes": 3,
    "label_map": {0: 0, 0.5: 1, 1: 2, 2: 2},
    "class_names": ["Normal", "Mild", "Moderate/Severe"],
    # data
    "processed_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\processed_dataset_128",
    "csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\12-data_leakage.csv",
    "weights_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\weights",
    # dataloader
    "batch_size": 4,
    "val_batch_size": 2,
    "test_batch_size": 2,
    "phase2_batch_size": 2,     # batch_size=1 breaks BatchNorm during fine-tuning
    "grad_accum_steps": 2,      # accumulate 2 steps → effective BS = 4
    "num_workers": 4,
    # model
    "dropout": 0.4,             # increased for regularization
    "label_smoothing": 0.1,     # soften overconfident predictions
    # phase 1 — head warmup (backbone frozen)
    "phase1_epochs": 15,
    "phase1_lr": 3e-4,
    # phase 2a — fine-tune (only layer4 unfrozen)
    "phase2a_epochs": 20,
    "phase2a_backbone_lr": 1e-5,
    "phase2a_head_lr": 5e-5,
    # phase 2b — fine-tune (layer3 + layer4 unfrozen)
    "phase2b_epochs": 30,
    "phase2b_backbone_lr": 5e-6,
    "phase2b_head_lr": 3e-5,
    # phase 3 — SWA
    "swa_epochs": 5,
    "swa_lr": 1e-5,
    # early stopping
    "patience": 20,
    # scheduler
    "lr_patience": 8,           # ReduceLROnPlateau patience
    "lr_factor": 0.5,           # LR reduction factor
    # weight decay
    "weight_decay": 5e-4,
    # TTA
    "tta_folds": 5,
    # MixUp
    "use_mixup": False,
    "mixup_alpha": 0.3,
    "mixup_start_epoch": 100,
    # model name
    "model_name": "ResNet50_MedicalNet_v3",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════
# SEED
# ══════════════════════════════════════════════════════════════
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# seed & device print moved into main() to avoid worker re-execution on Windows

# ══════════════════════════════════════════════════════════════
# AUGMENTATION
# ══════════════════════════════════════════════════════════════
def build_train_transforms():
    return Compose([
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.3, spatial_axis=1),
        RandFlip(prob=0.2, spatial_axis=2),
        RandAffine(
            prob=0.4,
            rotate_range=(0.10, 0.10, 0.10),
            scale_range=(0.08, 0.08, 0.08),
            mode="bilinear",
            padding_mode="zeros",
        ),
        RandGaussianNoise(prob=0.20, mean=0.0, std=0.03),
        RandScaleIntensity(factors=0.15, prob=0.3),
        RandAdjustContrast(prob=0.20, gamma=(0.90, 1.10)),
        RandCoarseDropout(
            holes=6,
            spatial_size=12,
            fill_value=0.0,
            prob=0.25,
        ),
    ])

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

# ══════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════
class BrainMRIDataset(Dataset):
    def __init__(self, root_dir: str, df: pd.DataFrame, augment: bool = False):
        self.root = root_dir
        self.augment = augment
        self.transforms = build_train_transforms() if augment else None
        self.label_map = CFG["label_map"]

        # keep only samples whose .npy exists
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

        if self.augment and self.transforms is not None:
            vol = vol[np.newaxis]                    # (1, D, H, W)
            vol = self.transforms(vol)
            if not isinstance(vol, np.ndarray):
                vol = vol.numpy()
            vol = vol.squeeze(0)
            vol = np.nan_to_num(vol, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32, copy=False)
            vol = np.clip(vol, -5.0, 5.0)

        tensor = torch.from_numpy(vol).unsqueeze(0)  # (1, D, H, W)
        return tensor, torch.tensor(label, dtype=torch.long)

# ══════════════════════════════════════════════════════════════
# MODEL — MedicalNet ResNet50 + Custom Head
# ══════════════════════════════════════════════════════════════
def _download_medicalnet_r50():
    from huggingface_hub import hf_hub_download
    print("MedicalNet ResNet50 weights downloading …")
    path = hf_hub_download(
        repo_id="TencentMedicalNet/MedicalNet-Resnet50",
        filename="resnet_50_23dataset.pth",
    )
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt if isinstance(ckpt, dict) and "state_dict" not in ckpt else ckpt.get("state_dict", ckpt)
    # strip 'module.' prefix
    clean = {k.replace("module.", ""): v for k, v in sd.items()}
    return clean


class BrainClassifier(nn.Module):
    """
    MedicalNet ResNet50 backbone -> custom head.

    Backbone outputs 2048-d features after global-average-pool.
    Head: FC 2048->512 -> BN -> GELU -> Dropout -> FC 512->128 -> BN -> GELU -> Dropout -> FC 128->C
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

        # expose backbone layers individually for selective freezing
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

    # ------------------------------------------------------------------
    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # last linear — small init for stable start
        last = self.head[-1]
        nn.init.normal_(last.weight, 0, 0.01)
        nn.init.zeros_(last.bias)

    # ------------------------------------------------------------------
    def freeze_backbone(self):
        for layer in [self.conv1, self.bn1, self.layer1,
                      self.layer2, self.layer3, self.layer4]:
            for p in layer.parameters():
                p.requires_grad = False
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🧊 Backbone frozen | trainable params: {t:,}")

    def unfreeze_from(self, stage: str = "layer3"):
        """Unfreeze backbone starting from *stage* onwards."""
        order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
        start = order.index(stage) if stage in order else 0
        for name in order[:start]:
            for p in getattr(self, name).parameters():
                p.requires_grad = False
        for name in order[start:]:
            for p in getattr(self, name).parameters():
                p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        a = sum(p.numel() for p in self.parameters())
        print(f"🔓 Unfrozen from {stage} | trainable: {t:,}/{a:,} ({100*t/a:.1f}%)")

    # ------------------------------------------------------------------
    def forward(self, x):
        x = self.maxpool(self.act(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.head(x)


# ══════════════════════════════════════════════════════════════
# MIXUP
# ══════════════════════════════════════════════════════════════
def mixup_data(x, y, alpha=0.3):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ══════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, n = 0.0, 0
    preds_all, labels_all = [], []
    for imgs, labs in loader:
        imgs, labs = imgs.to(device, non_blocking=True), labs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(imgs)
            loss = criterion(out, labs)
        loss_sum += loss.item() * labs.size(0)
        n += labs.size(0)
        preds_all.extend(out.argmax(1).cpu().numpy())
        labels_all.extend(labs.cpu().numpy())
    return {
        "loss": loss_sum / max(n, 1),
        "acc": 100 * accuracy_score(labels_all, preds_all),
        "bacc": 100 * balanced_accuracy_score(labels_all, preds_all),
        "preds": preds_all,
        "labels": labels_all,
    }


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    epoch_str="", accum_steps=1, use_mixup=False, mixup_alpha=0.3):
    model.train()
    loss_sum, n, correct, total = 0.0, 0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    loop = tqdm(loader, desc=epoch_str, leave=False)
    for step, (imgs, labs) in enumerate(loop):
        imgs = imgs.to(device, non_blocking=True)
        labs = labs.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            if use_mixup:
                mixed_imgs, targets_a, targets_b, lam = mixup_data(imgs, labs, mixup_alpha)
                out = model(mixed_imgs)
                loss = mixup_criterion(criterion, out, targets_a, targets_b, lam) / accum_steps
            else:
                out = model(imgs)
                loss = criterion(out, labs) / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # optimizer step every accum_steps or at last batch
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        bs = labs.size(0)
        loss_sum += loss.item() * accum_steps * bs
        n += bs
        # accuracy: for mixup use original labels
        with torch.no_grad():
            preds = out.argmax(1)
            if use_mixup:
                correct += (lam * (preds == targets_a).float() + (1 - lam) * (preds == targets_b).float()).sum().item()
            else:
                correct += (preds == labs).sum().item()
        total += bs

        loop.set_postfix(loss=f"{loss_sum/max(n,1):.4f}", acc=f"{100*correct/max(total,1):.1f}%")

    return loss_sum / max(n, 1), 100 * correct / max(total, 1)


# ══════════════════════════════════════════════════════════════
# TTA EVALUATION
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_tta(model, loader, device, n_folds=5):
    model.eval()
    transforms = build_tta_transforms()[:n_folds]
    all_preds, all_labels = [], []

    for imgs, labs in tqdm(loader, desc="TTA"):
        probs = torch.zeros(imgs.size(0), CFG["num_classes"], device=device)
        for tfm in transforms:
            if tfm is not None:
                batch = []
                for img in imgs:
                    a = tfm(img.numpy())
                    batch.append(torch.from_numpy(a) if isinstance(a, np.ndarray) else a)
                batch = torch.stack(batch).to(device)
            else:
                batch = imgs.to(device)
            probs += F.softmax(model(batch), dim=1)
        probs /= n_folds
        all_preds.extend(probs.argmax(1).cpu().numpy())
        all_labels.extend(labs.numpy())

    return all_preds, all_labels


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    seed_everything(CFG["seed"])
    print(f"Device: {DEVICE}")
    name = CFG["model_name"]

    # ── data split ───────────────────────────────────────────
    df = pd.read_csv(CFG["csv_path"])
    print(f"Total samples in CSV: {len(df)}")

    # CNN-only: keep one row per unique MR image (deduplicate)
    before = len(df)
    df = df.drop_duplicates(subset="path", keep="first").reset_index(drop=True)
    print(f"After MR dedup: {len(df)}  (removed {before - len(df)} duplicate MR rows)")

    sp1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tv_idx, te_idx = next(sp1.split(df, groups=df["OASISID"]))
    tv_df, te_df = df.iloc[tv_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)

    sp2 = GroupShuffleSplit(n_splits=1, test_size=15/85, random_state=42)
    tr_idx, va_idx = next(sp2.split(tv_df, groups=tv_df["OASISID"]))
    tr_df = tv_df.iloc[tr_idx].reset_index(drop=True)
    va_df = tv_df.iloc[va_idx].reset_index(drop=True)

    print(f"Train: {len(tr_df)} | Val: {len(va_df)} | Test: {len(te_df)}")

    # ── datasets ─────────────────────────────────────────────
    train_ds = BrainMRIDataset(CFG["processed_dir"], tr_df, augment=True)
    val_ds   = BrainMRIDataset(CFG["processed_dir"], va_df, augment=False)
    test_ds  = BrainMRIDataset(CFG["processed_dir"], te_df, augment=False)
    print(f"Train (valid npy): {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── sampler ──────────────────────────────────────────────
    targets = []
    for _, row in train_ds.df.iterrows():
        lv = float(row["memory"]) if not pd.isna(row["memory"]) else 0.0
        targets.append(CFG["label_map"].get(lv, 0))
    targets = np.array(targets)
    counts = np.bincount(targets, minlength=CFG["num_classes"])
    print(f"Class counts: {dict(zip(CFG['class_names'], counts))}")

    w = [1.0 / max(counts[t], 1) for t in targets]
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)

    # ── loaders ──────────────────────────────────────────────
    BS, NW = CFG["batch_size"], CFG["num_workers"]
    VAL_BS = CFG["val_batch_size"]
    TEST_BS = CFG["test_batch_size"]
    kw = dict(num_workers=NW, persistent_workers=True, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=BS, sampler=sampler, drop_last=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=VAL_BS, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=TEST_BS, shuffle=False, **kw)

    # ── model ────────────────────────────────────────────────
    model = BrainClassifier(
        num_classes=CFG["num_classes"],
        dropout=CFG["dropout"],
        pretrained=True,
    ).to(DEVICE)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_p:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    # history
    hist = {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": [], "va_bacc": []}

    best_bacc = 0.0
    best_acc  = 0.0
    best_loss = float("inf")
    patience_left = CFG["patience"]
    global_epoch = 0

    # ═══════════════════════════════════════════════════════════
    # PHASE 1 — HEAD WARMUP (backbone frozen)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🧊 PHASE 1: Head Warmup (backbone frozen)")
    print("=" * 60)
    model.freeze_backbone()

    opt1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["phase1_lr"], weight_decay=CFG["weight_decay"],
    )
    sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=CFG["phase1_epochs"], eta_min=1e-5)

    for ep in range(CFG["phase1_epochs"]):
        global_epoch += 1
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, opt1, scaler, DEVICE)
        val = evaluate(model, val_loader, criterion, DEVICE)
        sch1.step()
        lr = opt1.param_groups[0]["lr"]
        dt = (time.time() - t0) / 60

        hist["tr_loss"].append(tr_loss); hist["va_loss"].append(val["loss"])
        hist["tr_acc"].append(tr_acc);   hist["va_acc"].append(val["acc"])
        hist["va_bacc"].append(val["bacc"])

        improved = _save_if_better(model, name, val, best_bacc, best_acc, best_loss)
        if improved:
            best_bacc = max(best_bacc, val["bacc"])
            best_acc  = max(best_acc,  val["acc"])
            best_loss = min(best_loss, val["loss"])
            patience_left = CFG["patience"]
        else:
            patience_left -= 1

        print(f"[P1] Ep {global_epoch:3d} | "
              f"TrL {tr_loss:.4f} TrA {tr_acc:5.2f}% | "
              f"VaL {val['loss']:.4f} VaA {val['acc']:5.2f}% BAcc {val['bacc']:5.2f}% | "
              f"LR {lr:.2e} | {dt:.1f}m | pat={patience_left}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 2a — FINE-TUNE (only layer4 unfrozen)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🔓 PHASE 2a: Fine-tune (layer4 only)")
    print("=" * 60)
    model.unfreeze_from("layer4")

    # rebuild DataLoader with smaller batch for 8 GB GPU
    BS2 = CFG["phase2_batch_size"]
    ACCUM = CFG["grad_accum_steps"]
    train_loader_p2 = DataLoader(train_ds, batch_size=BS2, sampler=sampler, drop_last=True, **kw)
    print(f"   Phase 2 batch: {BS2} × {ACCUM} accum = {BS2*ACCUM} effective")

    def _build_param_groups(model, bb_lr, hd_lr):
        bb_params, hd_params = [], []
        for n, p in model.named_parameters():
            if p.requires_grad:
                (hd_params if "head" in n else bb_params).append(p)
        return [
            {"params": bb_params, "lr": bb_lr, "weight_decay": CFG["weight_decay"]},
            {"params": hd_params, "lr": hd_lr, "weight_decay": CFG["weight_decay"]},
        ]

    opt2a = optim.AdamW(_build_param_groups(
        model, CFG["phase2a_backbone_lr"], CFG["phase2a_head_lr"]
    ))
    sch2a = optim.lr_scheduler.ReduceLROnPlateau(
        opt2a, mode="max", factor=CFG["lr_factor"],
        patience=CFG["lr_patience"],
    )

    patience_left = CFG["patience"]  # reset patience for new phase

    for ep in range(CFG["phase2a_epochs"]):
        global_epoch += 1
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader_p2, criterion, opt2a, scaler, DEVICE, accum_steps=ACCUM)
        val = evaluate(model, val_loader, criterion, DEVICE)
        sch2a.step(val["bacc"])
        lrs = [pg["lr"] for pg in opt2a.param_groups]
        dt = (time.time() - t0) / 60

        hist["tr_loss"].append(tr_loss); hist["va_loss"].append(val["loss"])
        hist["tr_acc"].append(tr_acc);   hist["va_acc"].append(val["acc"])
        hist["va_bacc"].append(val["bacc"])

        improved = _save_if_better(model, name, val, best_bacc, best_acc, best_loss)
        if improved:
            best_bacc = max(best_bacc, val["bacc"])
            best_acc  = max(best_acc,  val["acc"])
            best_loss = min(best_loss, val["loss"])
            patience_left = CFG["patience"]
        else:
            patience_left -= 1

        print(f"[P2a] Ep {global_epoch:3d} | "
              f"TrL {tr_loss:.4f} TrA {tr_acc:5.2f}% | "
              f"VaL {val['loss']:.4f} VaA {val['acc']:5.2f}% BAcc {val['bacc']:5.2f}% | "
              f"bbLR {lrs[0]:.2e} hdLR {lrs[1]:.2e} | {dt:.1f}m | pat={patience_left}")

        if patience_left <= 0:
            print("⏹  Early stopping triggered in Phase 2a")
            break

    # ═══════════════════════════════════════════════════════════
    # PHASE 2b — FINE-TUNE (layer3 + layer4 unfrozen)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🔓 PHASE 2b: Fine-tune (layer3 + layer4)")
    print("=" * 60)
    model.unfreeze_from("layer3")

    opt2b = optim.AdamW(_build_param_groups(
        model, CFG["phase2b_backbone_lr"], CFG["phase2b_head_lr"]
    ))
    sch2b = optim.lr_scheduler.ReduceLROnPlateau(
        opt2b, mode="max", factor=CFG["lr_factor"],
        patience=CFG["lr_patience"],
    )

    patience_left = CFG["patience"]  # reset patience for new phase

    for ep in range(CFG["phase2b_epochs"]):
        global_epoch += 1
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader_p2, criterion, opt2b, scaler, DEVICE, accum_steps=ACCUM)
        val = evaluate(model, val_loader, criterion, DEVICE)
        sch2b.step(val["bacc"])
        lrs = [pg["lr"] for pg in opt2b.param_groups]
        dt = (time.time() - t0) / 60

        hist["tr_loss"].append(tr_loss); hist["va_loss"].append(val["loss"])
        hist["tr_acc"].append(tr_acc);   hist["va_acc"].append(val["acc"])
        hist["va_bacc"].append(val["bacc"])

        improved = _save_if_better(model, name, val, best_bacc, best_acc, best_loss)
        if improved:
            best_bacc = max(best_bacc, val["bacc"])
            best_acc  = max(best_acc,  val["acc"])
            best_loss = min(best_loss, val["loss"])
            patience_left = CFG["patience"]
        else:
            patience_left -= 1

        print(f"[P2b] Ep {global_epoch:3d} | "
              f"TrL {tr_loss:.4f} TrA {tr_acc:5.2f}% | "
              f"VaL {val['loss']:.4f} VaA {val['acc']:5.2f}% BAcc {val['bacc']:5.2f}% | "
              f"bbLR {lrs[0]:.2e} hdLR {lrs[1]:.2e} | {dt:.1f}m | pat={patience_left}")

        if patience_left <= 0:
            print("⏹  Early stopping triggered in Phase 2b")
            break

    # ═══════════════════════════════════════════════════════════
    # PHASE 3 — SWA (Stochastic Weight Averaging)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📊 PHASE 3: SWA (Stochastic Weight Averaging)")
    print("=" * 60)

    # reload best balanced-acc checkpoint before SWA
    best_path = os.path.join(CFG["weights_dir"], f"{name}_best_balanced.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        print(f"   Loaded best checkpoint ({best_bacc:.2f}% BAcc) for SWA start")

    model.unfreeze_from("layer4")

    # reuse smaller batch loader for SWA too
    train_loader_p3 = DataLoader(train_ds, batch_size=BS2, sampler=sampler, drop_last=True, **kw)

    opt3 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["swa_lr"], weight_decay=CFG["weight_decay"],
    )
    swa_model = AveragedModel(model)
    swa_sched = SWALR(opt3, swa_lr=CFG["swa_lr"])

    for ep in range(CFG["swa_epochs"]):
        global_epoch += 1
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader_p3, criterion, opt3, scaler, DEVICE, accum_steps=ACCUM)
        swa_model.update_parameters(model)
        swa_sched.step()
        dt = (time.time() - t0) / 60
        print(f"[SWA] Ep {global_epoch:3d} | TrL {tr_loss:.4f} TrA {tr_acc:5.2f}% | {dt:.1f}m")

    # update BN statistics for SWA model
    print("   Updating SWA batch-norm statistics …")
    torch.optim.swa_utils.update_bn(train_loader_p3, swa_model, device=DEVICE)

    # evaluate SWA model
    swa_val = evaluate(swa_model, val_loader, criterion, DEVICE)
    print(f"   SWA Val → Acc {swa_val['acc']:.2f}% | BAcc {swa_val['bacc']:.2f}%")

    # save SWA if it's better
    if swa_val["bacc"] >= best_bacc:
        torch.save(swa_model.module.state_dict(), os.path.join(CFG["weights_dir"], f"{name}_best_balanced.pth"))
        best_bacc = swa_val["bacc"]
        print(f"   ⭐ SWA improved BAcc → {best_bacc:.2f}%")
    torch.save(swa_model.module.state_dict(), os.path.join(CFG["weights_dir"], f"{name}_swa.pth"))

    # ═══════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════
    p1_end = CFG["phase1_epochs"]
    p2a_end = p1_end + CFG["phase2a_epochs"]
    _plot_curves(hist, name, p1_end, p2a_end)

    # ═══════════════════════════════════════════════════════════
    # TEST EVALUATION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🧪 TEST EVALUATION")
    print("=" * 60)

    # load best model
    model.load_state_dict(torch.load(os.path.join(CFG["weights_dir"], f"{name}_best_balanced.pth"), map_location=DEVICE))
    model.eval()

    # normal test
    normal = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\n── Normal Test ──")
    print(f"Accuracy: {normal['acc']:.2f}% | Balanced Accuracy: {normal['bacc']:.2f}%")
    print(classification_report(
        normal["labels"], normal["preds"], target_names=CFG["class_names"]
    ))

    # TTA test
    tta_preds, tta_labels = evaluate_tta(model, test_loader, DEVICE, CFG["tta_folds"])
    tta_bacc = 100 * balanced_accuracy_score(tta_labels, tta_preds)
    tta_acc  = 100 * accuracy_score(tta_labels, tta_preds)
    print(f"── TTA Test (x{CFG['tta_folds']}) ──")
    print(f"Accuracy: {tta_acc:.2f}% | Balanced Accuracy: {tta_bacc:.2f}%")
    print(classification_report(tta_labels, tta_preds, target_names=CFG["class_names"]))

    # confusion matrix
    use_tta = tta_bacc > normal["bacc"]
    tag = "TTA" if use_tta else "Normal"
    cm_preds = tta_preds if use_tta else normal["preds"]
    cm_labels = tta_labels if use_tta else normal["labels"]

    cm = confusion_matrix(cm_labels, cm_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CFG["class_names"],
                yticklabels=CFG["class_names"])
    plt.title(f"{name} — Confusion Matrix ({tag})")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{name}_confusion_matrix.png", dpi=150)
    plt.show()

    print(f"\n✅ Best result ({tag}): BAcc = {max(tta_bacc, normal['bacc']):.2f}%")


# ──────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────
def _save_if_better(model, name, val, best_bacc, best_acc, best_loss):
    wdir = CFG["weights_dir"]
    os.makedirs(wdir, exist_ok=True)
    improved = False
    if val["bacc"] > best_bacc:
        torch.save(model.state_dict(), os.path.join(wdir, f"{name}_best_balanced.pth"))
        print(f"   ⭐ Best BAcc: {val['bacc']:.2f}%")
        improved = True
    if val["acc"] > best_acc:
        torch.save(model.state_dict(), os.path.join(wdir, f"{name}_best_acc.pth"))
        print(f"   Best Acc: {val['acc']:.2f}%")
        improved = True
    if val["loss"] < best_loss:
        torch.save(model.state_dict(), os.path.join(wdir, f"{name}_best_loss.pth"))
        improved = True
    return improved


def _plot_curves(hist, name, p1_end, p2a_end=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax in axes:
        ax.axvline(p1_end, color="r", ls="--", alpha=.5, label="P2a (layer4)")
        if p2a_end is not None:
            ax.axvline(p2a_end, color="orange", ls="--", alpha=.5, label="P2b (layer3+4)")

    axes[0].plot(hist["tr_loss"], label="Train")
    axes[0].plot(hist["va_loss"], label="Val")
    axes[0].set(title=f"{name} — Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()

    axes[1].plot(hist["tr_acc"], label="Train")
    axes[1].plot(hist["va_acc"], label="Val")
    axes[1].set(title=f"{name} — Accuracy", xlabel="Epoch", ylabel="%")
    axes[1].legend()

    axes[2].plot(hist["va_bacc"], label="BAcc", color="green")
    axes[2].set(title=f"{name} — Balanced Accuracy", xlabel="Epoch", ylabel="%")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{name}_training_curves.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()


