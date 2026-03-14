"""
Sadece MLP ile klinik veri sınıflandırması
-CustomCNN.ipynb yapısına göre oluşturulmuştur.
MRI görüntüleri kullanılmaz, sadece klinik (tabular) veri kullanılır.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import time
import os

# ============================================================
# 1. SEED SABİTLEME
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed sabitlendi: {seed}")

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 2. DATASET - SADECE KLİNİK VERİ (MRI YOK)
# ============================================================
class ClinicalOnlyDataset(Dataset):
    """Sadece klinik (tabular) veri kullanan dataset (MRI yok)"""
    
    def __init__(self, data_csv, augment=False):
        self.data_csv = data_csv.reset_index(drop=True)
        self.augment = augment
        self.label_map = {0: 0, 0.5: 1, 1: 2, 2: 2}
        
        # Klinik sütunlar (OASISID, memory, path hariç)
        self.clinical_cols = [
            col for col in self.data_csv.columns
            if col not in ['OASISID', 'memory', 'path']
            and not str(col).startswith('Unnamed:')
        ]

    def __len__(self):
        return len(self.data_csv)
    
    def __getitem__(self, idx):
        row = self.data_csv.iloc[idx]
        
        # Label
        label_val = row['memory']
        try:
            label_val = float(label_val)
        except:
            label_val = 0.0
        label = self.label_map.get(label_val, 0)
        
        # Klinik veri
        clinical_series = row[self.clinical_cols]
        clinical_data = pd.to_numeric(clinical_series, errors='coerce').fillna(0).values.astype(np.float32)
        
        if self.augment:
             # Clinical Noise Injection (Gaussian Noise)
             noise = np.random.normal(0, 0.05, clinical_data.shape).astype(np.float32)
             clinical_data = clinical_data + noise

        return torch.tensor(clinical_data), torch.tensor(label, dtype=torch.long)

# ============================================================
# 3. SADECE MLP MODELİ (CNN YOK)
# ============================================================
class MLP_Only(nn.Module):
    """Geliştirilmiş MLP modeli - Residual bağlantılar ile"""
    
    def __init__(self, input_dim=170, num_classes=3):
        super(MLP_Only, self).__init__()

        # Block 1: input_dim -> 256
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # Block 2: 256 -> 128
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # Block 3: 128 -> 64
        self.block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
        )
        
        # Residual projection: 256 -> 64 (skip connection)
        self.residual_proj = nn.Linear(256, 64)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        x1 = self.block1(x)       # input_dim -> 256
        x2 = self.block2(x1)      # 256 -> 128
        x3 = self.block3(x2)      # 128 -> 64
        
        # Residual connection: block1 çıktısını block3 çıktısına ekle
        residual = self.residual_proj(x1)  # 256 -> 64
        x3 = x3 + residual
        
        out = self.classifier(x3)
        return out

# ============================================================
# 4. FOCAL LOSS
# ============================================================
# ============================================================
# 4. LOSS FONKSİYONLARI & YARDIMCI METOTLAR
# ============================================================
class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
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
    def __init__(self, ce_criterion, focal_criterion, warmup_epochs=10):
        super().__init__()
        self.ce_criterion = ce_criterion
        self.focal_criterion = focal_criterion
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        
    def forward(self, inputs, targets):
        if self.current_epoch < self.warmup_epochs:
            return self.ce_criterion(inputs, targets)
        else:
            return self.focal_criterion(inputs, targets)

def get_focal_gamma(epoch, warmup_epochs, max_gamma_epoch=50):
    """
    Focal Loss Gamma değerini dinamik olarak hesaplar.
    Warmup sonrası 1.0'dan başlayıp max_gamma_epoch'a kadar 5.0'a çıkar.
    """
    if epoch < warmup_epochs:
        return 0.0 # CE kullanılıyor
    
    if epoch >= max_gamma_epoch:
        return 5.0
    
    # Linear artış: 1.0 -> 5.0
    progress = (epoch - warmup_epochs) / (max_gamma_epoch - warmup_epochs)
    return 1.0 + (4.0 * progress)

# ============================================================
# 5. EĞİTİM FONKSİYONU
# ============================================================
def train_mlp_only(model, train_loader, val_loader, criterion, optimizer, device, 
                   epochs=100, patience=15, save_path="mlp_only_best.pth", scheduler=None):
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_bal_acc = 0.0
    patience_counter = patience

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    val_bal_accuracies = []

    for epoch in range(epochs):
        # Staged Loss & Gamma Update
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch)
            
            if epoch >= criterion.warmup_epochs:
                 # Gamma hesaplama
                 new_gamma = get_focal_gamma(epoch, criterion.warmup_epochs)
                 criterion.focal_criterion.gamma = new_gamma

            if epoch == criterion.warmup_epochs:
                print(f"\n🔄 FOCAL LOSS'A GEÇİŞ YAPILDI! (Epoch {epoch+1})\n")

        model.train()
        train_loss_sum = 0.0
        train_n = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)

        for batch_idx, (clinical, labels) in enumerate(loop):
            clinical = clinical.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(clinical)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
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
            for val_clinical, val_labels in val_loader:
                val_clinical = val_clinical.to(device, non_blocking=True)
                val_labels = val_labels.to(device, non_blocking=True).long()
                val_outputs = model(val_clinical)
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
        
        # Balanced Accuracy Hesapla
        avg_val_bal_acc = balanced_accuracy_score(all_val_labels, all_val_preds) * 100

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(avg_train_acc)
        val_accuracies.append(avg_val_acc)
        val_bal_accuracies.append(avg_val_bal_acc)

        elapsed = (time.time() - start_time)
        loss_type = "CE" if hasattr(criterion, 'current_epoch') and criterion.current_epoch < criterion.warmup_epochs else "FL"
        
        print(f"Epoch {epoch+1}/{epochs} [{loss_type}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}% | Val Bal Acc: {avg_val_bal_acc:.2f}% | "
              f"Süre: {elapsed:.2f} sn")
        
        # Early Stopping & Saving
        improved = False

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), "mlp_only_best_acc.pth")
            print(f" -> En iyi Accuracy bulundu: {best_val_acc:.2f}%")
            improved = True
            
        if avg_val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = avg_val_bal_acc
            torch.save(model.state_dict(), "mlp_only_best_bal_acc.pth")
            print(f" -> En iyi Balanced Accuracy bulundu: {best_val_bal_acc:.2f}%")
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
        
        torch.save(model.state_dict(), "data-mlp_only_last.pth")

    print(f"\n Eğitim tamamlandi. En iyi Loss: {best_val_loss:.4f} | En iyi Acc: {best_val_acc:.2f}%")
    
    # Grafikleri çiz
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Val Loss", marker="o")
    plt.title("MLP Only - Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label="Train Acc", marker="o")
    plt.plot(val_accuracies, label="Val Acc", marker="o")
    plt.title("MLP Only - Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(val_bal_accuracies, label="Val Bal Acc", marker="o", color='orange')
    plt.title("MLP Only - Balanced Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Balanced Acc (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("mlp_only_training_curves.png", dpi=150)
    plt.show()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# ============================================================
# 6. MAIN - ÇALIŞTIRMA
# ============================================================
if __name__ == "__main__":
    # Veri yükleme
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_csv = pd.read_csv(os.path.join(script_dir, "12-data_leakage.csv"))
    
    print(f"Toplam veri sayısı: {len(data_csv)}")
    
    # Klinik sütunları belirle
    clinical_cols = [
        col for col in data_csv.columns
        if col not in ['OASISID', 'memory', 'path']
        and not str(col).startswith('Unnamed:')
    ]
    print(f"Klinik özellik sayısı: {len(clinical_cols)}")
    
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
    
    # Klinik veri normalizasyonu
    for col in clinical_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        val_df[col] = pd.to_numeric(val_df[col], errors='coerce').fillna(0)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
    
    scaler = StandardScaler()
    train_df[clinical_cols] = scaler.fit_transform(train_df[clinical_cols])
    val_df[clinical_cols] = scaler.transform(val_df[clinical_cols])
    test_df[clinical_cols] = scaler.transform(test_df[clinical_cols])
    
    print("Klinik veriler normalize edildi.")
    
    # Dataset oluştur
    train_dataset = ClinicalOnlyDataset(train_df, augment=True) # Augmentasyon AÇIK
    val_dataset = ClinicalOnlyDataset(val_df, augment=False)
    test_dataset = ClinicalOnlyDataset(test_df, augment=False)
    
    print(f"\nTrain dataset samples: {len(train_dataset)}")
    print(f"Val dataset samples: {len(val_dataset)}")
    print(f"Test dataset samples: {len(test_dataset)}")
    
    # DataLoader
    BATCH_SIZE = 128
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model oluştur
    NUM_CLASSES = 3
    INPUT_DIM = len(clinical_cols)
    model = MLP_Only(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
    model.to(device)
    
    print(f"\nModel device: {next(model.parameters()).device}")
    print(f"Input dimension: {INPUT_DIM}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Class weights hesapla
    train_targets = []
    label_map = {0: 0, 0.5: 1, 1: 2, 2: 2}
    for _, row in train_df.iterrows():
        label_val = row['memory']
        try:
            label_val = float(label_val)
        except:
            label_val = 0.0
        label = label_map.get(label_val, 0)
        train_targets.append(label)

    train_targets = np.array(train_targets, dtype=np.int64)
    class_counts = np.bincount(train_targets, minlength=NUM_CLASSES)
    total = class_counts.sum()
    # Class weights hesapla (Sadece bilgi amaçlıydı, kafa karıştırmasın diye kaldırıldı)
    # Loss fonksiyonunda kullanılmıyor. Sadece Sampler kullanılıyor.
    
    # WeightedRandomSampler ile sınıf dengeleme
    sample_weights = 1.0 / class_counts[train_targets]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float64)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_targets), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    
    # Loss, Optimizer, Scheduler
    # Loss, Optimizer, Scheduler
    WARMUP_EPOCHS = 10
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    focal_criterion = BalancedFocalLoss(alpha=None, gamma=1.0, label_smoothing=0.1)
    criterion = StagedLoss(ce_criterion, focal_criterion, warmup_epochs=WARMUP_EPOCHS)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Eğitim başlat
    print("\n" + "="*60)
    print("SADECE MLP EĞİTİMİ BAŞLIYOR...")
    print("="*60 + "\n")
    
    train_mlp_only(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=40,
        patience=30,
        save_path="mlp_only_best.pth",
        scheduler=scheduler
    )
    
    # Test değerlendirmesi (ACCURACY MODEL)
    print("\n" + "="*60)
    print("EN İYİ ACCURACY MODELİ DEĞERLENDİRMESİ")
    print("="*60)
    
    model.load_state_dict(torch.load("mlp_only_best_acc.pth"))
    model.eval()
    
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from sklearn.preprocessing import label_binarize

    def evaluate_and_plot_cm(loader, set_name, save_suffix=""):
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for clinical, labels in tqdm(loader, desc=f"{set_name}"):
                clinical = clinical.to(device)
                outputs = model(clinical)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        print(f"\n{'='*40}")
        print(f"{set_name} Classification Report ({save_suffix}):")
        print("="*40)
        print(classification_report(all_labels, all_preds, target_names=['Normal', 'Mild', 'Moderate/Severe']))
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    # ACC Model için değerlendir
    train_labels, train_preds, _ = evaluate_and_plot_cm(train_loader, "TRAIN", "BestAcc")
    val_labels, val_preds, _ = evaluate_and_plot_cm(val_loader, "VALIDATION", "BestAcc")
    test_labels, test_preds, test_probs_acc = evaluate_and_plot_cm(test_loader, "TEST", "BestAcc")
    
    # 3'lü Confusion Matrix (ACC)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (labels, preds, title) in zip(axes, [
        (train_labels, train_preds, "Train"),
        (val_labels, val_preds, "Validation"),
        (test_labels, test_preds, "Test")
    ]):
        cm = confusion_matrix(labels, preds)
        acc = accuracy_score(labels, preds)
        bal_acc = balanced_accuracy_score(labels, preds)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Mild', 'Moderate/Severe'],
                    yticklabels=['Normal', 'Mild', 'Moderate/Severe'])
        ax.set_title(f'{title} CM (Best Acc)\nAcc: {acc*100:.2f}% | Bal Acc: {bal_acc*100:.2f}%')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig("mlp_only_cm_best_acc.png", dpi=150)
    print("\n✅ Best Acc confusion matrix kaydedildi: mlp_only_cm_best_acc.png")
    plt.show() # Show Acc plot

    # ----------------------------------------------------------------
    # Test değerlendirmesi (LOSS MODEL) - KULLANICININ İSTEĞİ
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("EN İYİ LOSS MODELİ DEĞERLENDİRMESİ")
    print("="*60)

    try:
        model.load_state_dict(torch.load("mlp_only_best.pth")) # Best Loss Model
        model.eval()

        # Loss Model için değerlendir
        test_labels_loss, test_preds_loss, test_probs_loss = evaluate_and_plot_cm(test_loader, "TEST", "BestLoss")

        # Confusion Matrix (LOSS)
        plt.figure(figsize=(6, 5))
        cm_loss = confusion_matrix(test_labels_loss, test_preds_loss)
        acc_loss = accuracy_score(test_labels_loss, test_preds_loss)
        bal_acc_loss = balanced_accuracy_score(test_labels_loss, test_preds_loss)
        
        sns.heatmap(cm_loss, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Normal', 'Mild', 'Moderate/Severe'],
                    yticklabels=['Normal', 'Mild', 'Moderate/Severe'])
        plt.title(f'Test Confusion Matrix (Best Loss Model)\nAcc: {acc_loss*100:.2f}% | Bal Acc: {bal_acc_loss*100:.2f}%')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig("mlp_only_cm_best_loss.png", dpi=150)
        print("\n✅ Best Loss confusion matrix kaydedildi: mlp_only_cm_best_loss.png")
        plt.show()

    except Exception as e:
        print(f"\n⚠️ Best Loss modeli yüklenirken hata oluştu: {e}")

    # ----------------------------------------------------------------
    # Test değerlendirmesi (BALANCED ACC MODEL) - YENİ ÖZELLİK
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("EN İYİ BALANCED ACCURACY MODELİ DEĞERLENDİRMESİ")
    print("="*60)

    try:
        model.load_state_dict(torch.load("mlp_only_best_bal_acc.pth")) # Best Bal Acc Model
        model.eval()

        # Bal Acc Model için değerlendir
        test_labels_bal, test_preds_bal, test_probs_bal = evaluate_and_plot_cm(test_loader, "TEST", "BestBalAcc")

        # Confusion Matrix (BAL ACC)
        plt.figure(figsize=(6, 5))
        cm_bal = confusion_matrix(test_labels_bal, test_preds_bal)
        acc_bal = accuracy_score(test_labels_bal, test_preds_bal)
        bal_acc_bal = balanced_accuracy_score(test_labels_bal, test_preds_bal)
        
        sns.heatmap(cm_bal, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=['Normal', 'Mild', 'Moderate/Severe'],
                    yticklabels=['Normal', 'Mild', 'Moderate/Severe'])
        plt.title(f'Test Confusion Matrix (Best Bal Acc Model)\nAcc: {acc_bal*100:.2f}% | Bal Acc: {bal_acc_bal*100:.2f}%')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig("mlp_only_cm_best_bal_acc.png", dpi=150)
        print("\n✅ Best Bal Acc confusion matrix kaydedildi: mlp_only_cm_best_bal_acc.png")
        plt.show()

        # --- ROC AUC CURVE FOR BEST BAL ACC MODEL ---
        def plot_roc_curve(labels, probs, title, filename):
            y_bin = label_binarize(labels, classes=[0, 1, 2])
            n_classes = y_bin.shape[1]
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            class_names = ['Normal', 'Mild', 'Moderate/Severe']
            colors = ['#2196F3', '#FF9800', '#E91E63'] # Blue, Orange, Pink
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                
            fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), probs.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            plt.figure(figsize=(8, 6))
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label='{0} (AUC = {1:0.3f})'.format(class_names[i], roc_auc[i]))
            
            plt.plot(fpr["micro"], tpr["micro"], color='black', lw=2, linestyle='--',
                     label='Micro-avg (AUC = {0:0.3f})'.format(roc_auc["micro"]))
            
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
            plt.show()
            print(f"✅ ROC Curve kaydedildi: {filename}")

        plot_roc_curve(test_labels_bal, test_probs_bal, 'Test ROC Curves (Best Bal Acc Model)', 'mlp_only_roc_best_bal_acc.png')

    except Exception as e:
        print(f"\n⚠️ Best Bal Acc modeli yüklenirken hata oluştu: {e}")

    # ----------------------------------------------------------------
    # FEATURE IMPORTANCE ANALİZİ
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALİZİ YAPILIYOR...")
    print("="*60)
    
    def analyze_feature_importance(model, loader, device, feature_names):
        model.eval()
        criterion = nn.CrossEntropyLoss()
        
        # 1. Tüm veriyi topla
        all_clinical = []
        all_labels = []
        for clinical, labels in loader:
            all_clinical.append(clinical)
            all_labels.append(labels)
            
        all_clinical = torch.cat(all_clinical).to(device)
        all_labels = torch.cat(all_labels).to(device).long()
        
        # 2. Baseline Loss hesapla
        with torch.no_grad():
            outputs = model(all_clinical)
            baseline_loss = criterion(outputs, all_labels).item()
        
        print(f"Baseline Loss: {baseline_loss:.6f}")
        
        feature_importance = []
        
        # 3. Her özellik için Permutation Importance
        for i, feature_name in enumerate(tqdm(feature_names, desc="Analyzing Features")):
            # Orijinal veriyi kopyala
            shuffled_clinical = all_clinical.clone()
            
            # İlgili sütunu karıştır (shuffle)
            perm = torch.randperm(shuffled_clinical.size(0))
            shuffled_clinical[:, i] = shuffled_clinical[perm, i]
            
            # Yeni loss hesapla
            with torch.no_grad():
                outputs = model(shuffled_clinical)
                loss = criterion(outputs, all_labels).item()
            
            # Loss farkı = Önem derecesi
            importance = loss - baseline_loss
            feature_importance.append({
                'Feature': feature_name,
                'Importance': importance
            })
            
        return pd.DataFrame(feature_importance).sort_values(by='Importance', ascending=False)

    try:
        # En iyi model ile analiz yap (Genelde Best Bal Acc veya Best Loss iyidir)
        model.load_state_dict(torch.load("mlp_only_best_bal_acc.pth"))
        
        # Test seti üzerinde analiz yap
        df_importance = analyze_feature_importance(model, test_loader, device, clinical_cols)
        
        # Kaydet
        df_importance.to_csv("mlp_feature_importance.csv", index=False)
        print("\n✅ Feature importance sonuçları kaydedildi: mlp_feature_importance.csv")
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=df_importance.head(20), palette='viridis')
        plt.title('Top 20 Most Important Features (MLP)')
        plt.tight_layout()
        plt.savefig("mlp_feature_importance.png", dpi=150)
        plt.show()
        print("✅ Feature importance grafiği kaydedildi: mlp_feature_importance.png")
        
    except Exception as e:
        print(f"\n⚠️ Feature Importance analizi sırasında hata: {e}")


