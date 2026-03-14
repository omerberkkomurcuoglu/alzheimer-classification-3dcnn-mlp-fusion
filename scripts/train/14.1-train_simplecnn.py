"""
Custom 3D CNN ile MRI görüntü sınıflandırması
Basitleştirilmiş pipeline: CrossEntropyLoss + label_smoothing + güçlü augmentasyon
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
    RandScaleIntensity, RandGaussianSmooth, RandAdjustContrast,
)
from mri_pipeline import (
    coerce_memory_label,
    filter_existing_mri_rows,
    load_sanitized_mri_volume,
    normalize_mri_volume,
    sanitize_mri_volume,
)

# ============================================================
# 1. SEED SABİTLEME
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # Sabit input boyutunda hız optimizasyonu
    print(f"Seed sabitlendi: {seed}")

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 2. FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    """Focal Loss — kolay örnekleri baskılar, zor örneklere odaklanır."""
    def __init__(self, gamma=1.5, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================
# 3. DATASET + AUGMENTATION
# ============================================================
def get_train_transforms():
    return Compose([
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandFlip(prob=0.5, spatial_axis=2),       # Z ekseni de ekle
        RandAffine(
            prob=0.5,
            rotate_range=(0.15, 0.15, 0.15),      # 0.1 → 0.15
            scale_range=(0.15, 0.15, 0.15),
            translate_range=(5, 5, 5),             # Translation ekle
            mode='bilinear',
            padding_mode='zeros'
        ),
        RandGaussianNoise(prob=0.2, mean=0.0, std=0.02),  # Noise açıldı
        RandScaleIntensity(factors=0.15, prob=0.5),
        RandAdjustContrast(prob=0.3, gamma=(0.85, 1.15)),
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
# 6. EĞİTİM FONKSİYONU
# ============================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                epochs=100, patience=30, save_path="simplecnn_best.pth", scheduler=None,
                model_name="Custom3DCNN", step_scheduler_per_batch=False):
    # Mixed Precision Training (AMP) - ~2x hızlanma
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None
    use_amp = device.type == 'cuda'
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_balanced_acc = 0.0
    patience_counter = patience

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
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
            
            # AMP: Forward pass float16 ile (hız artışı)
            with torch.amp.autocast(device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Accuracy hesabı
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
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

            # OneCycleLR: her batch sonunda step
            if step_scheduler_per_batch and scheduler:
                scheduler.step()

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

        # Epoch-level scheduler (OneCycleLR değilse)
        if scheduler and not step_scheduler_per_batch:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(avg_train_acc)
        val_accuracies.append(avg_val_acc)

        elapsed = (time.time() - start_time) / 60
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}% | "
              f"Balanced Acc: {balanced_acc:.2f}% | "
              f"LR: {current_lr:.6f} | Süre: {elapsed:.2f} dk")
        
        # Early Stopping & Saving
        improved = False

        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            torch.save(model.state_dict(), f"{model_name}_best_balanced.pth")
            print(f" -> ⭐ En iyi Balanced Accuracy bulundu: {best_balanced_acc:.2f}%")
            improved = True
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), f"{model_name}_best_acc.pth")
            print(f" -> En iyi Accuracy bulundu: {best_val_acc:.2f}%")
            improved = True
             
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f" -> En iyi Loss bulundu: {best_val_loss:.4f}")
            improved = True
        
        if improved:
            patience_counter = patience
        else:
            patience_counter -= 1
            print(f"Bir gelişme yok ({patience_counter} patience kaldı)")
            
            if patience_counter == 0:
                print("Erken durdurma (Early Stopping)")
                break
        
        torch.save(model.state_dict(), f"{model_name}_last.pth")

    print(f"\n Eğitim tamamlandı. En iyi Loss: {best_val_loss:.4f} | En iyi Acc: {best_val_acc:.2f}% | En iyi Balanced Acc: {best_balanced_acc:.2f}%")
    
    # Grafikleri çiz
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Val Loss", marker="o")
    plt.title(f"{model_name} - Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Acc", marker="o")
    plt.plot(val_accuracies, label="Val Acc", marker="o")
    plt.title(f"{model_name} - Accuracy Curve")
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
    
    # CNN-only: aynı MR görüntüsünü birden fazla kullanma (deduplicate)
    before = len(data_csv)
    data_csv = data_csv.drop_duplicates(subset="path", keep="first").reset_index(drop=True)
    print(f"After MR dedup: {len(data_csv)}  (removed {before - len(data_csv)} duplicate MR rows)")
    
    MODEL_NAME = "Custom3DCNN"
    print(f"Toplam veri sayısı: {len(data_csv)}")
    
    # Veri bölme (70% Train, 15% Val, 15% Test)
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
    print(f"Class distribution: Normal={class_counts[0]}, Hafif={class_counts[1]}, Orta/Ağır={class_counts[2]}")
    
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
    
    # Model oluştur (Harici dosyadan)
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from custom_model import Custom3DCNN
    
    # Dropout hafif artırıldı - daha iyi regularizasyon
    model = Custom3DCNN(num_classes=NUM_CLASSES, dropout=0.3)
    model.to(device)
    
    print(f"\nModel: {MODEL_NAME} (Geliştirilmiş Mimari)")
    print(f"Model device: {next(model.parameters()).device}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam parametre: {total_params:,}")
    print(f"Eğitilebilir parametre: {trainable_params:,}")
    
    # =============================================
    # LOSS FONKSİYONU — CrossEntropyLoss + Class Weights
    # =============================================
    # Class weights: azınlık sınıflarına daha yüksek ağırlık
    class_weights_raw = 1.0 / (class_counts.astype(np.float64) + 1)
    class_weights_raw = class_weights_raw / class_weights_raw.sum() * NUM_CLASSES
    class_weights_tensor = torch.tensor(class_weights_raw, dtype=torch.float32).to(device)
    print(f"\nClass weights: {class_weights_raw}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    
    print(f"\n📋 Loss Stratejisi:")
    print(f"   - CrossEntropyLoss (label_smoothing=0.1, class_weights)")
    print(f"   - Class balance: WeightedRandomSampler + Class Weights")
    print(f"   - Early stopping: BALANCED ACCURACY ile")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    
    # OneCycleLR — warm-up dahil, en stabil scheduler
    total_epochs = 150
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,
        steps_per_epoch=len(train_loader),
        epochs=total_epochs,
        pct_start=0.1,       # İlk %10 warm-up
        anneal_strategy='cos',
        div_factor=10,       # Başlangıç LR = max_lr/10 = 3e-4
        final_div_factor=100 # Son LR = max_lr/1000 = 3e-6
    )
    
    # Eğitim başlat
    print("\n" + "="*60)
    print(f"🚀 {MODEL_NAME} EĞİTİMİ BAŞLIYOR...")
    print(f"   Max LR: 3e-3 | Epochs: {total_epochs} | Patience: 35")
    print(f"   Scheduler: OneCycleLR (warm-up %10, cosine annealing)")
    print(f"   Loss: CrossEntropyLoss + class_weights + label_smoothing")
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
        model_name=MODEL_NAME,
        step_scheduler_per_batch=True,  # OneCycleLR her batch'te step atar
    )
    # ============================================================
    # TEST DEĞERLENDİRMESİ (TTA ile)
    # ============================================================
    print("\n" + "="*60)
    print("TEST DEĞERLENDİRMESİ (TTA: Test-Time Augmentation)")
    print("="*60)
    
    model.load_state_dict(torch.load(f"{MODEL_NAME}_best_balanced.pth"))
    model.eval()
    
    def predict_with_tta(model, images):
        """TTA: orijinal + 3 eksen flip → ortalama softmax"""
        predictions = []
        with torch.no_grad():
            # Orijinal
            predictions.append(F.softmax(model(images), dim=1))
            # Flip: D ekseni
            predictions.append(F.softmax(model(torch.flip(images, [2])), dim=1))
            # Flip: H ekseni
            predictions.append(F.softmax(model(torch.flip(images, [3])), dim=1))
            # Flip: W ekseni
            predictions.append(F.softmax(model(torch.flip(images, [4])), dim=1))
        return torch.stack(predictions).mean(dim=0)
    
    # --- Normal test (TTA'sız) ---
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
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Hafif', 'Orta/Ağır']))
    
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
    print(classification_report(tta_labels, tta_preds, target_names=['Normal', 'Hafif', 'Orta/Ağır']))
    
    # Confusion Matrix (TTA sonuçları ile)
    cm = confusion_matrix(tta_labels, tta_preds)
    test_acc = tta_acc
    test_bacc = tta_bacc
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Hafif', 'Orta/Ağır'],
                yticklabels=['Normal', 'Hafif', 'Orta/Ağır'])
    plt.title(f'TEST - Confusion Matrix (TTA)\nAcc: {test_acc:.2f}% | Balanced Acc: {test_bacc:.2f}%')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(f"{MODEL_NAME}_confusion_matrix_tta.png", dpi=150)
    plt.show()
