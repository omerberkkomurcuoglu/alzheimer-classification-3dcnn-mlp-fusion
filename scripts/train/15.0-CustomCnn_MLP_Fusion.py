"""
Custom3DCNN + MLP Fusion Model (v5 - CE Only)
- Pretrained weights (Custom3DCNN + MLP)
- Backbone ALWAYS frozen
- CrossEntropy Loss only (label smoothing=0.1)
- Modality Dropout (70% clinical / 30% MRI drop)
- Deep Supervision (CNN aux + MLP aux)
- CosineAnnealingWarmRestarts scheduler
- Cross-Modal Attention Gates
- Equal-dim Feature Projection + L2 Normalization
- Asymmetric Modality Dropout
- Deep Supervision (CNN & MLP aux heads)
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import time
import sys

from monai.transforms import (
    Compose, RandFlip, RandAffine, RandScaleIntensity, RandAdjustContrast, RandGaussianNoise
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
    torch.backends.cudnn.benchmark = True

# ============================================================
# 2. LOSS FONKSİYONU
# ============================================================
# Sadece CrossEntropy Loss kullanılıyor (label smoothing=0.1)
# Focal Loss kaldırıldı: WeightedRandomSampler sınıf dengelemeyi zaten sağlıyor.

# ============================================================
# 3. DATASET & AUGMENTATION
# ============================================================
def get_train_transforms():
    return Compose([
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.3, spatial_axis=1),
        RandAffine(
            prob=0.5,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1),
            mode='bilinear',
            padding_mode='zeros'
        ),
        RandScaleIntensity(factors=0.1, prob=0.5),
        RandAdjustContrast(prob=0.3, gamma=(0.9, 1.1)),
        RandGaussianNoise(prob=0.1, std=0.05),
    ])

class MRIClinicalDataset(Dataset):
    def __init__(self, root_dir, data_csv, clinical_cols, augment=False):
        self.root_dir = root_dir
        self.data_csv = data_csv.reset_index(drop=True)
        self.clinical_cols = clinical_cols
        self.augment = augment
        self.label_map = {0: 0, 0.5: 1, 1: 2, 2: 2}
        self.train_transforms = get_train_transforms() if augment else None
        
        self.data_csv = filter_existing_mri_rows(root_dir, self.data_csv)

    def __len__(self):
        return len(self.data_csv)
    
    def __getitem__(self, idx):
        row = self.data_csv.iloc[idx]
        session_id = row["path"]
        file_path = os.path.join(self.root_dir, f"{session_id}.npy")
        
        data = load_sanitized_mri_volume(file_path)
        label = coerce_memory_label(row['memory'])
        
        clinical_series = row[self.clinical_cols]
        clinical_data = pd.to_numeric(clinical_series, errors='coerce').fillna(0).values.astype(np.float32)
        
        if self.augment and self.train_transforms is not None:
            data = data[np.newaxis, ...]
            data = self.train_transforms(data)
            if isinstance(data, np.ndarray):
                data = data.squeeze(0)
            else:
                data = data.squeeze(0).numpy()
            data = sanitize_mri_volume(data)
            
            # Clinical Noise Injection
            noise = np.random.normal(0, 0.05, clinical_data.shape).astype(np.float32)
            clinical_data = clinical_data + noise

        data = normalize_mri_volume(data)
        mri_tensor = torch.from_numpy(data).unsqueeze(0)
        clinical_tensor = torch.tensor(clinical_data)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return mri_tensor, clinical_tensor, label_tensor

# ============================================================
# 4. FUSION MODEL
# ============================================================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from custom_model import Custom3DCNN
except ImportError:
    pass

class CNN_MLP_Fusion(nn.Module):
    def __init__(self, clinical_dim=72, num_classes=3, dropout=0.3,
                 pretrained_cnn_path=None, pretrained_mlp_path=None,
                 modality_dropout_rate=0.3):
        super().__init__()
        
        self.modality_dropout_rate = modality_dropout_rate
        
        # === CNN Branch ===
        print("Initializing Custom3DCNN...")
        cnn_full = Custom3DCNN(num_classes=num_classes, dropout=dropout)
        
        if pretrained_cnn_path and os.path.exists(pretrained_cnn_path):
            print(f"Loading CNN weights from {pretrained_cnn_path}...")
            state_dict = torch.load(pretrained_cnn_path, map_location='cpu')
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            cnn_full.load_state_dict(state_dict, strict=True)
            print("✅ CNN weights loaded.")
        else:
            print("⚠️ Pretrained CNN weights not found.")

        self.input_norm = cnn_full.input_norm
        self.stem = cnn_full.stem
        self.layer1 = cnn_full.layer1
        self.layer2 = cnn_full.layer2
        self.se2 = cnn_full.se2
        self.layer3 = cnn_full.layer3
        self.se3 = cnn_full.se3
        self.gap = cnn_full.gap
        
        # --- CNN Auxiliary Classifier ---
        self.cnn_aux = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        # === MLP Branch ===
        self.mlp_block1 = nn.Sequential(
            nn.Linear(clinical_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.mlp_block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.mlp_block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )
        self.mlp_residual_proj = nn.Linear(256, 64)
        
        # --- MLP Auxiliary Classifier ---
        self.mlp_aux = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

        if pretrained_mlp_path and os.path.exists(pretrained_mlp_path):
            print(f"Loading MLP weights from {pretrained_mlp_path}...")
            mlp_state = torch.load(pretrained_mlp_path, map_location='cpu')
            
            new_state_dict = {}
            for k, v in mlp_state.items():
                if k.startswith('block1'):
                    new_state_dict[k.replace('block1', 'mlp_block1')] = v
                elif k.startswith('block2'):
                    new_state_dict[k.replace('block2', 'mlp_block2')] = v
                elif k.startswith('block3'):
                    new_state_dict[k.replace('block3', 'mlp_block3')] = v
                elif k.startswith('residual_proj'):
                    new_state_dict[k.replace('residual_proj', 'mlp_residual_proj')] = v
            
            missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
            print("✅ MLP weights loaded (partial).")
        else:
            print("⚠️ Pretrained MLP weights not found.")
        
        # === Feature Projection (Equal Dim) ===
        self.cnn_proj = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mlp_proj = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # === Cross-Modal Attention Gates ===
        total_dim = 256  # 128 + 128
        self.cnn_gate = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.Sigmoid()
        )
        self.mlp_gate = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.Sigmoid()
        )
        
        # === Final Classifier ===
        self.fusion_classifier = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )
        
        nn.init.normal_(self.fusion_classifier[-1].weight, mean=0, std=0.01)
        nn.init.zeros_(self.fusion_classifier[-1].bias)
    
    def forward(self, mri, clinical):
        # CNN feature extraction
        x = self.input_norm(mri)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        x = self.se3(x)
        x = self.gap(x)
        cnn_features = x.view(x.size(0), -1)  # (B, 256)
        
        # MLP feature extraction
        m1 = self.mlp_block1(clinical)
        m2 = self.mlp_block2(m1)
        m3 = self.mlp_block3(m2)
        mlp_features = m3 + self.mlp_residual_proj(m1)  # (B, 64)
        
        # Auxiliary Outputs
        cnn_aux_out = self.cnn_aux(cnn_features)
        mlp_aux_out = self.mlp_aux(mlp_features)

        # Asymmetric Modality Dropout
        if self.training and self.modality_dropout_rate > 0:
            r = torch.rand(1).item()
            if r < self.modality_dropout_rate:
                if torch.rand(1).item() < 0.7:  # %70 klinik drop (Klinik veriye zorla bağımlılığı kır)
                    mlp_features = torch.zeros_like(mlp_features)
                else:  # %30 MRI drop
                    cnn_features = torch.zeros_like(cnn_features)
        
        # Feature Projection & L2 Normalization
        cnn_proj = self.cnn_proj(cnn_features)
        mlp_proj = self.mlp_proj(mlp_features)
        cnn_proj = F.normalize(cnn_proj, p=2, dim=1)
        mlp_proj = F.normalize(mlp_proj, p=2, dim=1)
        
        # Cross-Modal Attention Fusion
        combined = torch.cat([cnn_proj, mlp_proj], dim=1)  # (B, 256)
        cnn_attention = self.cnn_gate(combined)
        mlp_attention = self.mlp_gate(combined)
        gated_cnn = cnn_proj * cnn_attention
        gated_mlp = mlp_proj * mlp_attention
        gated = torch.cat([gated_cnn, gated_mlp], dim=1)  # (B, 256)
        
        output = self.fusion_classifier(gated)
        
        if self.training:
            return output, cnn_aux_out, mlp_aux_out
        else:
            return output

# ============================================================
# 5. EĞİTİM
# ============================================================
def train_fusion(model, train_loader, val_loader, criterion,
                 optimizer, device, epochs=100, patience=25,
                 scheduler=None, model_name="CNN_MLP_Fusion", 
                 save_dir="."):
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    use_amp = device.type == 'cuda'
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_balanced_acc = 0.0
    patience_counter = patience

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    bal_acc_history = []

    # Backbone'ları ilk 5 epoch dondur
    print("\n🔒 INITIALIZATION: BACKBONE FROZEN (İlk 5 epoch)")
    print("Eğitilen katmanlar: Projections, Gates, Aux Heads, Fusion Classifier")
    print("5. Epoch sonrasında CNN ve MLP backboneları Fine-Tuning için açılacaktır!")
    
    trainable_keywords = ["fusion_classifier", "cnn_gate", "mlp_gate", 
                          "cnn_proj", "mlp_proj", "cnn_aux", "mlp_aux"]
    
    trainable_params = 0
    frozen_params = 0
    for name, param in model.named_parameters():
        if any(k in name for k in trainable_keywords):
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    print(f"   Trainable: {trainable_params:,} params")
    print(f"   Frozen: {frozen_params:,} params")

    for epoch in range(epochs):
        loss_tag = "CE"
        
        # 5. Epoch'ta kilitleri açarak Fine-tuning başlatıyoruz
        if epoch == 5:
            print("\n🔓 Epoch 5: Unfreezing CNN & MLP backbones! Fine-tuning begins...")
            for param in model.stem.parameters(): param.requires_grad = True
            for param in model.layer1.parameters(): param.requires_grad = True
            for param in model.layer2.parameters(): param.requires_grad = True
            for param in model.layer3.parameters(): param.requires_grad = True
            for param in model.mlp_block1.parameters(): param.requires_grad = True
            for param in model.mlp_block2.parameters(): param.requires_grad = True
            
            # Öğrenme oranını düşür
            for g in optimizer.param_groups:
                g['lr'] = 1e-4

        model.train()
        train_loss_sum = 0
        train_n = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)

        for batch_idx, (mri, clinical, labels) in enumerate(loop):
            mri = mri.to(device, non_blocking=True)
            clinical = clinical.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            

            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs, cnn_aux, mlp_aux = model(mri, clinical)
                
                loss_main = criterion(outputs, labels)
                loss_cnn = criterion(cnn_aux, labels)
                loss_mlp = criterion(mlp_aux, labels)
                
                # Deep Supervision: CNN aux ağırlığı yüksek
                loss = loss_main + 1.0 * loss_cnn + 0.2 * loss_mlp
            
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
            
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
            loop.set_postfix({'loss': f"{avg_loss:.4f}", 'acc': f"{acc:.1f}%"})
        
        avg_train_loss = train_loss_sum / max(train_n, 1)
        avg_train_acc = 100 * correct / total

        # Validation (her zaman CE ile değerlendir, karşılaştırılabilirlik için)
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for val_mri, val_clinical, val_labels in val_loader:
                val_mri = val_mri.to(device, non_blocking=True)
                val_clinical = val_clinical.to(device, non_blocking=True)
                val_labels = val_labels.to(device, non_blocking=True).long()
                
                val_outputs = model(val_mri, val_clinical)
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

        # Scheduler Step
        if scheduler:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(avg_train_acc)
        val_accuracies.append(avg_val_acc)
        bal_acc_history.append(balanced_acc)

        elapsed = (time.time() - start_time) / 60
        
        print(f"Epoch {epoch+1}/{epochs} [{loss_tag}] | "
              f"Train: {avg_train_loss:.4f} / {avg_train_acc:.1f}% | "
              f"Val: {avg_val_loss:.4f} / {avg_val_acc:.1f}% | "
              f"BalAcc: {balanced_acc:.1f}% | "
              f"LR: {current_lr:.6f} | {elapsed:.1f}dk")

        improved = False

        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_best_balanced.pth"))
            print(f" -> ⭐ En iyi Balanced Accuracy: {best_balanced_acc:.2f}%")
            improved = True
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_best_acc.pth"))
            print(f" -> En iyi Accuracy: {best_val_acc:.2f}%")
            improved = True
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_best.pth"))
            print(f" -> En iyi Loss: {best_val_loss:.4f}")
            improved = True
        
        if improved:
            patience_counter = patience
            print(f"   -> Patience resetlendi ({patience})")
        else:
            patience_counter -= 1
            print(f"Bir gelişme yok ({patience_counter} patience kaldı)")
            
            if patience_counter == 0:
                print("Erken durdurma (Early Stopping)")
                break
    
        torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_last.pth"))

    print(f"\n✅ Eğitim tamamlandı.")
    print(f"   En iyi Loss: {best_val_loss:.4f}")
    print(f"   En iyi Acc: {best_val_acc:.2f}%")
    print(f"   En iyi Balanced Acc: {best_balanced_acc:.2f}%")
    
    # Grafik (3 panel)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    axes[0].plot(train_losses, label="Train Loss", marker="o", markersize=2)
    axes[0].plot(val_losses, label="Val Loss", marker="o", markersize=2)
    axes[0].set_title(f"{model_name} - Kayıp")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(train_accuracies, label="Train Acc", marker="o", markersize=2)
    axes[1].plot(val_accuracies, label="Val Acc", marker="o", markersize=2)
    axes[1].set_title(f"{model_name} - Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    axes[2].plot(bal_acc_history, label="Balanced Acc", marker="o", markersize=2, color='orange')
    axes[2].set_title(f"{model_name} - Balanced Accuracy")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("Balanced Accuracy (%)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_training_curves.png"), dpi=150)
    plt.show()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# ============================================================
# 6. MAIN
# ============================================================
if __name__ == "__main__":
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Seed sabitlendi: 42")
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    processed_dir = "C:\\Users\\berkk\\Desktop\\Bitirme_Projesi\\processed_dataset_128"
    data_csv = pd.read_csv(os.path.join(script_dir, "12-data_leakage.csv"))
    
    cnn_weights = os.path.join(weights_dir, "Custom3DCNN_best_balanced.pth")
    mlp_weights = os.path.join(weights_dir, "mlp_only_best_bal_acc.pth")
    
    MODEL_NAME = "Fusion_Full"
    
    print(f"Toplam veri sayısı: {len(data_csv)}")
    clinical_cols = [col for col in data_csv.columns if col not in ['OASISID', 'memory', 'path']]
    CLINICAL_DIM = len(clinical_cols)
    print(f"Klinik özellik sayısı: {CLINICAL_DIM}")
    
    # Veri bölme (70/15/15)
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_inds, test_inds = next(splitter_test.split(data_csv, groups=data_csv['OASISID']))
    train_val_df = data_csv.iloc[train_val_inds].reset_index(drop=True)
    test_df = data_csv.iloc[test_inds].reset_index(drop=True)

    val_ratio = 15 / 85
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_inds, val_inds = next(splitter_val.split(train_val_df, groups=train_val_df['OASISID']))
    train_df = train_val_df.iloc[train_inds].reset_index(drop=True)
    val_df = train_val_df.iloc[val_inds].reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Normalizasyon
    scaler = StandardScaler()
    train_df[clinical_cols] = scaler.fit_transform(train_df[clinical_cols])
    val_df[clinical_cols] = scaler.transform(val_df[clinical_cols])
    test_df[clinical_cols] = scaler.transform(test_df[clinical_cols])
    print("✅ Klinik veriler normalize edildi.")
    
    # Datasets
    train_dataset = MRIClinicalDataset(processed_dir, train_df, clinical_cols, augment=True)
    val_dataset = MRIClinicalDataset(processed_dir, val_df, clinical_cols, augment=False)
    test_dataset = MRIClinicalDataset(processed_dir, test_df, clinical_cols, augment=False)
    
    # Sampler
    label_map = {0: 0, 0.5: 1, 1: 2, 2: 2}
    NUM_CLASSES = 3
    
    filtered_train_df = train_dataset.data_csv
    train_targets = []
    for _, row in filtered_train_df.iterrows():
        label = label_map.get(float(row['memory']), 0)
        train_targets.append(label)
    
    train_targets = np.array(train_targets, dtype=np.int64)
    class_counts = np.bincount(train_targets, minlength=NUM_CLASSES)
    print(f"\nSınıf dağılımı: Normal={class_counts[0]}, Hafif={class_counts[1]}, Orta/Ağır={class_counts[2]}")
    
    # Class weights (inverse frequency)
    total_samples = len(train_targets)
    class_weights = torch.tensor(
        [total_samples / (NUM_CLASSES * count) for count in class_counts],
        dtype=torch.float32
    ).to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(2)}")
    
    sample_weights = torch.tensor([1.0 / (class_counts[t] + 1) for t in train_targets], dtype=torch.float64)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # DataLoaders
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)
    
    # Model
    print("\n" + "="*60)
    print("MODEL INITIALIZATION (Fusion v3 - Frozen Backbone)")
    print("="*60)
    
    model = CNN_MLP_Fusion(
        clinical_dim=CLINICAL_DIM,
        num_classes=NUM_CLASSES,
        dropout=0.3,
        pretrained_cnn_path=cnn_weights,
        pretrained_mlp_path=mlp_weights,
        modality_dropout_rate=0.5 # Dropout artırıldı
    )
    model.to(device)
    
    # Loss fonksiyonu (Sınıf Ağırlıkları eklendi)
    class_weights = torch.tensor([1.0, 2.5, 2.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Optimizer - sadece trainable params
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-3, weight_decay=1e-4
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    
    TOTAL_EPOCHS = 100
    
    print("\n" + "="*60)
    print("🚀 EĞİTİM BAŞLIYOR...")
    print(f"   Pretrained CNN: {os.path.basename(cnn_weights)}")
    print(f"   Pretrained MLP: {os.path.basename(mlp_weights)}")
    print(f"   Backbone: İlk 5 Epoch FROZEN, Sonra UNFREEZE (Fine-Tuning)")
    print(f"   Loss: CrossEntropy (Ağırlıklı: [1.0, 2.5, 2.0], label_smoothing=0.1)")
    print(f"   Class Balance: WeightedRandomSampler + Class Weights")
    print(f"   Modality Dropout: 50% (70% klinik / 30% MRI drop)")
    print(f"   LR: 1e-3 + CosineAnnealingWarmRestarts (T_0=15, T_mult=2)")
    print("="*60 + "\n")
    
    train_fusion(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=TOTAL_EPOCHS,
        patience=25,
        scheduler=scheduler,
        model_name=MODEL_NAME,
        save_dir=weights_dir
    )
    
    # --- TEST DEĞERLENDİRMESİ ---
    print("\n" + "="*60)
    print("TEST EVALUATION")
    print("="*60)
    
    best_model_path = os.path.join(weights_dir, f"{MODEL_NAME}_best_balanced.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("✅ Best Balanced Model loaded.")
    else:
        print("⚠️ Best model not found, using last model.")
    
    model.eval()
    
    # --- TTA (Test-Time Augmentation) ---
    TTA_AUGMENTS = 5  # original + 4 augmented
    print(f"TTA aktif: {TTA_AUGMENTS}x augmentation")
    
    all_probs = []  # softmax probabilities
    all_labels = []
    
    with torch.no_grad():
        for mri, clinical, labels in tqdm(test_loader, desc="Test (TTA)"):
            mri = mri.to(device)
            clinical = clinical.to(device)
            
            # TTA: orijinal + flip'ler + noise
            tta_outputs = []
            
            # 1. Orijinal
            out = model(mri, clinical)
            tta_outputs.append(F.softmax(out, dim=1))
            
            # 2. Flip axis 2 (sagittal)
            out = model(torch.flip(mri, [2]), clinical)
            tta_outputs.append(F.softmax(out, dim=1))
            
            # 3. Flip axis 3 (coronal)
            out = model(torch.flip(mri, [3]), clinical)
            tta_outputs.append(F.softmax(out, dim=1))
            
            # 4. Flip axis 4 (axial)
            out = model(torch.flip(mri, [4]), clinical)
            tta_outputs.append(F.softmax(out, dim=1))
            
            # 5. Gaussian noise
            noisy_mri = mri + torch.randn_like(mri) * 0.02
            out = model(noisy_mri, clinical)
            tta_outputs.append(F.softmax(out, dim=1))
            
            # Ortalama
            avg_probs = torch.stack(tta_outputs).mean(dim=0)
            all_probs.append(avg_probs.cpu())
            all_labels.extend(labels.numpy())
    
    all_probs = torch.cat(all_probs, dim=0)
    all_preds = torch.argmax(all_probs, dim=1).numpy()
    
    print("\n📊 Classification Report (TTA):")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Mild', 'Moderate/Severe']))
    
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    test_acc = 100 * np.mean(np.array(all_labels) == all_preds)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Balanced Accuracy: {bal_acc*100:.2f}%")
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Mild', 'Moderate/Severe'],
                yticklabels=['Normal', 'Mild', 'Moderate/Severe'])
    plt.title(f'{MODEL_NAME} - Test Confusion Matrix (TTA x{TTA_AUGMENTS})\nAcc: {test_acc:.2f}% | Balanced Acc: {bal_acc*100:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(weights_dir, f"{MODEL_NAME}_test_confusion_matrix.png"), dpi=150)
    plt.show()
