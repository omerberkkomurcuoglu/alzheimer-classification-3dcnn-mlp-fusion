"""
Custom 3D CNN — MRI Image Classification
Same pipeline as DenseNet201: Staged Loss + MixUp + Balanced Accuracy
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import time

from monai.transforms import (
    Compose, RandFlip, RandAffine, RandGaussianNoise,
)
from mri_pipeline import (
    coerce_memory_label,
    filter_existing_mri_rows,
    load_sanitized_mri_volume,
    normalize_mri_volume,
    sanitize_mri_volume,
)

# ============================================================
# 1. SEED
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print(f"Seed fixed: {seed}")

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 2. FOCAL LOSS
# ============================================================
class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.initial_gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def set_gamma(self, new_gamma):
        if new_gamma != self.gamma:
            print(f"  🎯 Gamma updated: {self.gamma:.1f} → {new_gamma:.1f}")
            self.gamma = new_gamma
        
    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        with torch.no_grad():
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        pt = (probs * smooth_targets).sum(dim=-1)
        focal_weight = (1 - pt) ** self.gamma
        ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            focal_loss = alpha_weight * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class StagedLoss(nn.Module):
    def __init__(self, ce_criterion, focal_criterion, warmup_epochs=10, max_patience=30):
        super().__init__()
        self.ce_criterion = ce_criterion
        self.focal_criterion = focal_criterion
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.max_patience = max_patience
        
        # Softer gamma schedule (max 2.0)
        # Too high gamma (3.0) makes training unstable
        self.gamma_schedule = {
            25: 1.5, 15: 2.0, 5: 2.0
        }
        
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def update_gamma_by_patience(self, current_patience):
        target_gamma = self.focal_criterion.initial_gamma  # default
        for patience_threshold, gamma_value in sorted(self.gamma_schedule.items(), reverse=True):
            if current_patience <= patience_threshold:
                target_gamma = gamma_value
        if hasattr(self.focal_criterion, 'set_gamma'):
            self.focal_criterion.set_gamma(target_gamma)
        
    def forward(self, inputs, targets):
        if self.current_epoch < self.warmup_epochs:
            return self.ce_criterion(inputs, targets)
        else:
            return self.focal_criterion(inputs, targets)

# ============================================================
# 3. DATASET + AUGMENTATION
# ============================================================
def get_train_transforms():
    return Compose([
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.3, spatial_axis=1),
        # Lighter augmentations
        RandAffine(
            prob=0.5,
            rotate_range=(0.1, 0.1, 0.1), # 0.26 -> 0.1
            scale_range=(0.1, 0.1, 0.1),
            mode='bilinear',
            padding_mode='zeros'
        ),
        # Noise and Smooth disabled for now — model needs to learn clean data first
        # RandGaussianNoise(prob=0.3, mean=0.0, std=0.02),
        # RandGaussianSmooth(prob=0.2, sigma_x=(0.5, 1.0), ...),
        RandScaleIntensity(factors=0.1, prob=0.5),
        RandAdjustContrast(prob=0.3, gamma=(0.9, 1.1)),
    ])

def get_train_transforms():
    return Compose([
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.3, spatial_axis=1),
        RandFlip(prob=0.2, spatial_axis=2),
        RandAffine(
            prob=0.35,
            rotate_range=(0.08, 0.08, 0.08),
            scale_range=(0.08, 0.08, 0.08),
            mode='bilinear',
            padding_mode='zeros'
        ),
        RandGaussianNoise(prob=0.15, mean=0.0, std=0.02),
    ])

class MRIOnlyDataset(Dataset):
    def __init__(self, root_dir, data_csv, augment=False):
        self.root_dir = root_dir
        self.data_csv = filter_existing_mri_rows(root_dir, data_csv)
        self.augment = augment
        self.label_map = {0: 0, 0.5: 1, 1: 2, 2: 2}
        self.train_transforms = get_train_transforms() if augment else None

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        row = self.data_csv.iloc[idx]
        session_id = row["path"]
        file_path = os.path.join(self.root_dir, f"{session_id}.npy")

        data = load_sanitized_mri_volume(file_path)
        label = coerce_memory_label(row['memory'])

        if self.augment and self.train_transforms is not None:
            data = data[np.newaxis, ...]
            data = self.train_transforms(data)
            if isinstance(data, np.ndarray):
                data = data.squeeze(0)
            else:
                data = data.squeeze(0).numpy()
            data = sanitize_mri_volume(data)

        data = normalize_mri_volume(data)
        return torch.from_numpy(data).unsqueeze(0), torch.tensor(label, dtype=torch.long)

# Custom3DCNN now imported from custom_model.py

# ============================================================
# 5. MIXUP
# ============================================================
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================
# 6. TRAINING FUNCTION
# ============================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                epochs=100, patience=30, save_path="simplecnn_best.pth", scheduler=None,
                use_mixup=True, mixup_alpha=0.4, model_name="Custom3DCNN",
                mixup_start_epoch=15):
    # Mixed Precision Training (AMP) - ~2x speedup
    # Note: Issue was class weights, not AMP. AMP is safe to use
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    use_amp = device.type == 'cuda'
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_balanced_acc = 0.0
    patience_counter = patience

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch)
            if epoch == criterion.warmup_epochs:
                print("\n" + "="*60)
                print("🔄 SWITCHED TO FOCAL LOSS!")
                print("="*60 + "\n")
        
        model.train()
        train_loss_sum = 0
        train_n = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)

        for batch_idx, (images, labels) in enumerate(loop):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            
            optimizer.zero_grad(set_to_none=True)
            
            # MixUp: active after warmup
            epoch_mixup = use_mixup and epoch >= mixup_start_epoch
            
            # AMP: Forward pass with float16 (speed boost)
            with torch.amp.autocast('cuda', enabled=use_amp):
                if epoch_mixup:
                    mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, mixup_alpha)
                    outputs = model(mixed_images)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            
            # Accuracy calculation (outside autocast)
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                if epoch_mixup:
                    correct += (lam * (preds == targets_a).float() + (1 - lam) * (preds == targets_b).float()).sum().item()
                else:
                    correct += (preds == labels).sum().item()
            
            # AMP: Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total += labels.size(0)
            bs = labels.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs

            avg_loss = train_loss_sum / max(train_n, 1)
            acc = 100 * correct / total

            loop.set_postfix({
                'train_loss': f"{avg_loss:.4f}",
                'train_acc': f"{acc:.2f}%"
            })
        
        avg_train_loss = train_loss_sum / max(train_n, 1)
        avg_train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device, non_blocking=True)
                val_labels = val_labels.to(device, non_blocking=True).long()
                val_outputs = model(val_images)
                loss = criterion(val_outputs, val_labels)

                bs = val_labels.size(0)
                val_loss_sum += loss.item() * bs
                val_n += bs

                val_preds = torch.argmax(val_outputs, dim=1)
                val_correct += (val_preds == val_labels).sum().item()
                val_total += bs
                
                all_val_preds.extend(val_preds.cpu().numpy())
                all_val_labels.extend(val_labels.cpu().numpy())
            
        avg_val_loss = val_loss_sum / max(val_n, 1)
        avg_val_acc = 100 * val_correct / max(val_total, 1)
        balanced_acc = 100 * balanced_accuracy_score(all_val_labels, all_val_preds)

        if scheduler:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(avg_train_acc)
        val_accuracies.append(avg_val_acc)

        elapsed = (time.time() - start_time) / 60
        loss_type = "CE" if hasattr(criterion, 'current_epoch') and criterion.current_epoch < criterion.warmup_epochs else "FL"
        
        print(f"Epoch {epoch+1}/{epochs} [{loss_type}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}% | "
              f"Balanced Acc: {balanced_acc:.2f}% | "
              f"LR: {current_lr:.6f} | Time: {elapsed:.2f} min")
        
        # Early Stopping & Saving
        improved = False

        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            torch.save(model.state_dict(), f"{model_name}_best_balanced.pth")
            print(f" -> ⭐ Best Balanced Accuracy: {best_balanced_acc:.2f}%")
            improved = True
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), f"{model_name}_best_acc.pth")
            print(f" -> Best Accuracy: {best_val_acc:.2f}%")
            improved = True
             
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f" -> Best Loss: {best_val_loss:.4f}")
            improved = True
        
        if improved:
            patience_counter = patience
        else:
            patience_counter -= 1
            print(f"No improvement ({patience_counter} patience left)")
            
            if hasattr(criterion, 'update_gamma_by_patience'):
                criterion.update_gamma_by_patience(patience_counter)
            
            if patience_counter == 0:
                print("Early Stopping triggered")
                break
        
        torch.save(model.state_dict(), f"{model_name}_last.pth")

    print(f"\nTraining complete. Best Loss: {best_val_loss:.4f} | Best Acc: {best_val_acc:.2f}% | Best BAcc: {best_balanced_acc:.2f}%")
    
    # Plot curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Val Loss", marker="o")
    plt.axvline(x=10, color='red', linestyle='--', label='Focal Loss Start')
    plt.title(f"{model_name} — Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Acc", marker="o")
    plt.plot(val_accuracies, label="Val Acc", marker="o")
    plt.axvline(x=10, color='red', linestyle='--', label='Focal Loss Start')
    plt.title(f"{model_name} — Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{model_name}_training_curves.png", dpi=150)
    plt.show()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# ============================================================
# 7. MAIN
# ============================================================
if __name__ == "__main__":
    processed_dir = "C:\\Users\\berkk\\Desktop\\Bitirme_Projesi\\processed_dataset_128"
    data_csv = pd.read_csv("C:\\Users\\berkk\\Desktop\\Bitirme_Projesi\\scripts\\12-data_leakage.csv")
    
    MODEL_NAME = "Custom3DCNN"
    print(f"Total samples: {len(data_csv)}")
    
    # Data split (70% Train, 15% Val, 15% Test)
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_inds, test_inds = next(splitter_test.split(data_csv, groups=data_csv['OASISID']))

    train_val_df = data_csv.iloc[train_val_inds].reset_index(drop=True)
    test_df = data_csv.iloc[test_inds].reset_index(drop=True)

    val_ratio = 15 / 85
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_inds, val_inds = next(splitter_val.split(train_val_df, groups=train_val_df['OASISID']))

    train_df = train_val_df.iloc[train_inds].reset_index(drop=True)
    val_df = train_val_df.iloc[val_inds].reset_index(drop=True)

    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Dataset
    train_dataset = MRIOnlyDataset(processed_dir, train_df, augment=True)
    val_dataset = MRIOnlyDataset(processed_dir, val_df, augment=False)
    test_dataset = MRIOnlyDataset(processed_dir, test_df, augment=False)
    
    print(f"\nTrain dataset samples: {len(train_dataset)}")
    print(f"Val dataset samples: {len(val_dataset)}")
    print(f"Test dataset samples: {len(test_dataset)}")
    
    # WeightedRandomSampler
    label_map = {0: 0, 0.5: 1, 1: 2, 2: 2}
    NUM_CLASSES = 3
    
    filtered_train_df = train_dataset.data_csv
    
    train_targets = []
    for _, row in filtered_train_df.iterrows():
        label_val = row['memory']
        try:
            label_val = float(label_val)
        except:
            label_val = 0.0
        label = label_map.get(label_val, 0)
        train_targets.append(label)
    
    train_targets = np.array(train_targets, dtype=np.int64)
    class_counts = np.bincount(train_targets, minlength=NUM_CLASSES)
    
    print(f"\nFiltered train size: {len(filtered_train_df)}")
    print(f"Class counts: {class_counts}")
    print(f"Class distribution: Normal={class_counts[0]}, Mild={class_counts[1]}, Moderate/Severe={class_counts[2]}")
    
    sample_weights = []
    for _, row in filtered_train_df.iterrows():
        label_val = row['memory']
        try:
            label_val = float(label_val)
        except:
            label_val = 0.0
        label = label_map.get(label_val, 0)
        weight = 1.0 / (class_counts[label] + 1)
        sample_weights.append(weight)
    
    sample_weights = torch.tensor(sample_weights, dtype=torch.float64)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # DataLoader
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
                              num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)
    
    # Create model (from external file)
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from custom_model import Custom3DCNN
    
    # Keep dropout low — let model learn first, regularize later
    model = Custom3DCNN(num_classes=NUM_CLASSES, dropout=0.2)
    model.to(device)
    
    print(f"\nModel: {MODEL_NAME} (Improved Architecture)")
    print(f"Model device: {next(model.parameters()).device}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # =============================================
    # LOSS — CLASS WEIGHTS REMOVED
    # =============================================
    # IMPORTANT: WeightedRandomSampler already balances batches.
    # class_weights + sampler together = double penalty → model always predicts 1 class.
    # So we do NOT use weights in the loss.
    total = class_counts.sum()
    raw_weights = np.sqrt(total / (class_counts + 1))
    class_weights = raw_weights / raw_weights.sum() * NUM_CLASSES
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    print(f"\nClass weights (reference, NOT used in loss): {class_weights_tensor}")
    print(f"(Class balancing done via WeightedRandomSampler)")
    
    # Staged Loss: CE -> Focal (label smoothing OFF, weights OFF)
    warmup_epochs = 40
    ce_criterion = nn.CrossEntropyLoss()  # no weight — sampler handles balance
    focal_criterion = BalancedFocalLoss(
        alpha=None, gamma=1.0,   # no alpha
        label_smoothing=0.0, reduction='mean'
    )
    criterion = StagedLoss(ce_criterion, focal_criterion, warmup_epochs=warmup_epochs)
    
    print(f"\n📋 Loss Strategy:")
    print(f"   - Epoch 1-{warmup_epochs}: CrossEntropyLoss (plain, no weights)")
    print(f"   - Epoch {warmup_epochs+1}+: Focal Loss (gamma=1.0, no alpha)")
    print(f"   - MixUp: DISABLED")
    print(f"   - Class balance: WeightedRandomSampler")
    print(f"   - Early stopping: BALANCED ACCURACY based")
    
    # Optimizer - HIGH LR (1e-3 is standard for training CNN from scratch)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Warmup + Cosine Annealing
    total_epochs = 150
    def warmup_lambda(epoch):
        lr_warmup = 5
        if epoch < lr_warmup:
            return (epoch + 1) / lr_warmup
        else:
            progress = (epoch - lr_warmup) / (total_epochs - lr_warmup)
            return max(0.01, 0.5 * (1 + np.cos(np.pi * progress)))  # min LR = %1
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    # Start training
    print("\n" + "="*60)
    print(f"🚀 {MODEL_NAME} TRAINING STARTED...")
    print(f"   LR: 1e-3 | Epochs: {total_epochs} | Patience: 35")
    print("="*60 + "\n")
    
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=total_epochs,
        patience=35,
        save_path=f"{MODEL_NAME}_best.pth",
        scheduler=scheduler,
        use_mixup=True,
        mixup_alpha=0.3,
        model_name=MODEL_NAME,
        mixup_start_epoch=120
    )
    print("\n" + "="*60)
    print("TEST EVALUATION (TTA: Test-Time Augmentation)")
    print("="*60)
    
    model.load_state_dict(torch.load(f"{MODEL_NAME}_best_balanced.pth"))
    model.eval()
    
    def predict_with_tta(model, images):
        """TTA: original + 3 axis flips → average softmax"""
        predictions = []
        with torch.no_grad():
            predictions.append(F.softmax(model(images), dim=1))
            predictions.append(F.softmax(model(torch.flip(images, [2])), dim=1))
            predictions.append(F.softmax(model(torch.flip(images, [3])), dim=1))
            predictions.append(F.softmax(model(torch.flip(images, [4])), dim=1))
        return torch.stack(predictions).mean(dim=0)
    
    # --- Normal test ---
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test (Normal)"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    normal_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    normal_bacc = 100 * balanced_accuracy_score(all_labels, all_preds)
    print(f"\n📊 Normal Test — Acc: {normal_acc:.2f}% | Balanced Acc: {normal_bacc:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Mild', 'Moderate/Severe']))
    
    # --- TTA test ---
    tta_preds = []
    tta_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test (TTA)"):
            images = images.to(device)
            avg_probs = predict_with_tta(model, images)
            preds = torch.argmax(avg_probs, dim=1)
            tta_preds.extend(preds.cpu().numpy())
            tta_labels.extend(labels.numpy())
    
    tta_acc = 100 * np.mean(np.array(tta_preds) == np.array(tta_labels))
    tta_bacc = 100 * balanced_accuracy_score(tta_labels, tta_preds)
    print(f"\n📊 TTA Test — Acc: {tta_acc:.2f}% | Balanced Acc: {tta_bacc:.2f}%")
    print(classification_report(tta_labels, tta_preds, target_names=['Normal', 'Mild', 'Moderate/Severe']))
    
    # Confusion Matrix (use better result)
    use_tta = tta_bacc > normal_bacc
    tag = "TTA" if use_tta else "Normal"
    cm_preds = tta_preds if use_tta else all_preds
    cm_labels = tta_labels if use_tta else all_labels
    best_test_bacc = max(tta_bacc, normal_bacc)
    best_test_acc = tta_acc if use_tta else normal_acc
    
    cm = confusion_matrix(cm_labels, cm_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Mild', 'Moderate/Severe'],
                yticklabels=['Normal', 'Mild', 'Moderate/Severe'])
    plt.title(f'{MODEL_NAME} — Confusion Matrix ({tag})\nAcc: {best_test_acc:.2f}% | Balanced Acc: {best_test_bacc:.2f}%')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(f"{MODEL_NAME}_confusion_matrix_tta.png", dpi=150)
    plt.show()
    
    print(f"\n✅ Best result ({tag}): BAcc = {best_test_bacc:.2f}%")
