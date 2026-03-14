"""
CNN_MLP_Fusion Test Script
Eğitilmiş Fusion modeli ile Validation + Test değerlendirmesi.
Confusion Matrix + AUC-ROC Eğrileri
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from mri_pipeline import coerce_memory_label, filter_existing_mri_rows, load_normalized_mri_volume

# ============================================================
# 1. DATASET
# ============================================================
class MRIClinicalDataset(Dataset):
    """MRI + Klinik veri dataset"""
    
    def __init__(self, root_dir, data_csv, clinical_cols):
        self.root_dir = root_dir
        self.data_csv = data_csv.reset_index(drop=True)
        self.clinical_cols = clinical_cols
        self.label_map = {0: 0, 0.5: 1, 1: 2, 2: 2}
        
        self.data_csv = filter_existing_mri_rows(root_dir, self.data_csv)

    def __len__(self):
        return len(self.data_csv)
    
    def __getitem__(self, idx):
        row = self.data_csv.iloc[idx]
        session_id = row["path"]
        file_path = os.path.join(self.root_dir, f"{session_id}.npy")
        
        data = load_normalized_mri_volume(file_path)
        label = coerce_memory_label(row['memory'])
        
        clinical_series = row[self.clinical_cols]
        clinical_data = pd.to_numeric(clinical_series, errors='coerce').fillna(0).values.astype(np.float32)
        
        mri_tensor = torch.from_numpy(data).unsqueeze(0)
        clinical_tensor = torch.tensor(clinical_data)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return mri_tensor, clinical_tensor, label_tensor

# ============================================================
# 2. MODEL IMPORT
# ============================================================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Manuel import yerine doğrudan model tanımla
from custom_model import Custom3DCNN

class CNN_MLP_Fusion(nn.Module):
    """
    Custom3DCNN + MLP Fusion (v3 - Cross-Modal Attention)
    
    MRI → Custom3DCNN → 256 → proj(128) ─┐
                                            ├─ Cross-Modal Attention → FC → 3 sınıf
    Klinik → MLP → 64 → proj(128) ────────┘
    """
    def __init__(self, clinical_dim=72, num_classes=3, dropout=0.3,
                 modality_dropout_rate=0.3):
        super().__init__()
        
        self.modality_dropout_rate = modality_dropout_rate
        
        # === CNN Branch ===
        cnn_full = Custom3DCNN(num_classes=num_classes, dropout=dropout)
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
        import torch.nn.functional as F
        
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
        
        # Classification
        output = self.fusion_classifier(gated)
        return output

# ============================================================
# 3. DEĞERLENDİRME FONKSİYONU
# ============================================================
def evaluate_set(model, loader, device, set_name, class_names, save_prefix):
    """Confusion Matrix + AUC-ROC"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for mri, clinical, labels in tqdm(loader, desc=f"{set_name}"):
            mri = mri.to(device)
            clinical = clinical.to(device)
            outputs = model(mri, clinical)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    num_classes = len(class_names)
    
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    
    print(f"\n{'='*60}")
    print(f"{set_name} RESULTS (3 CLASSES)")
    print(f"{'='*60}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Balanced Accuracy: {bal_acc*100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{set_name} - Confusion Matrix\nAcc: {acc*100:.2f}% | Balanced Acc: {bal_acc*100:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_confusion_matrix.png", dpi=150)
    print(f"✅ Confusion Matrix kaydedildi: {save_prefix}_confusion_matrix.png")
    
    # AUC-ROC
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
    
    fpr_micro, tpr_micro, _ = roc_curve(labels_bin.ravel(), all_probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, color='black', lw=2, linestyle='--',
             label=f'Micro-avg (AUC = {roc_auc_micro:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{set_name} - ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_roc_curve.png", dpi=150)
    print(f"✅ ROC Eğrisi kaydedildi: {save_prefix}_roc_curve.png")
    
    print(f"\nAUC Scores:")
    for name, score in auc_scores.items():
        print(f"  {name}: {score:.3f}")
    print(f"  Micro-avg: {roc_auc_micro:.3f}")
    
    return acc, bal_acc, auc_scores

# ============================================================
# 4. MAIN
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    processed_dir = "C:\\Users\\berkk\\Desktop\\Bitirme_Projesi\\processed_dataset_128"
    data_csv = pd.read_csv("C:\\Users\\berkk\\Desktop\\Bitirme_Projesi\\scripts\\12-data_leakage.csv")
    
    clinical_cols = [col for col in data_csv.columns if col not in ['OASISID', 'memory', 'path']]
    CLINICAL_DIM = len(clinical_cols)
    print(f"Toplam veri: {len(data_csv)} | Klinik özellik: {CLINICAL_DIM}")
    
    # Veri bölme (aynı split)
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_inds, test_inds = next(splitter_test.split(data_csv, groups=data_csv['OASISID']))

    train_val_df = data_csv.iloc[train_val_inds].reset_index(drop=True)
    test_df = data_csv.iloc[test_inds].reset_index(drop=True)

    val_ratio = 15 / 85
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_inds, val_inds = next(splitter_val.split(train_val_df, groups=train_val_df['OASISID']))
    
    train_df = train_val_df.iloc[train_inds].reset_index(drop=True)
    val_df = train_val_df.iloc[val_inds].reset_index(drop=True)
    
    # Klinik veri normalizasyonu (train ile aynı scaler!)
    for col in clinical_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        val_df[col] = pd.to_numeric(val_df[col], errors='coerce').fillna(0)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
    
    scaler = StandardScaler()
    scaler.fit(train_df[clinical_cols])
    val_df[clinical_cols] = scaler.transform(val_df[clinical_cols])
    test_df[clinical_cols] = scaler.transform(test_df[clinical_cols])
    
    # Datasetler
    val_dataset = MRIClinicalDataset(processed_dir, val_df, clinical_cols)
    test_dataset = MRIClinicalDataset(processed_dir, test_df, clinical_cols)
    
    print(f"Validation: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    BATCH_SIZE = 10
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    model = CNN_MLP_Fusion(clinical_dim=CLINICAL_DIM, num_classes=3, dropout=0.3)
    model.to(device)
    
    weights_path = "C:\\Users\\berkk\\Desktop\\Bitirme_Projesi\\scripts\\weights\\Fusion_Full_best_balanced.pth"
    
    if not os.path.exists(weights_path):
        print(f"\n❌ {weights_path} bulunamadı!")
        print("Lütfen önce '17-densenet_mlp_fusion.py' ile modeli eğitin.")
        exit()
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"\n✅ Model yüklendi: {weights_path}")
    except RuntimeError as e:
        print(f"\n❌ Model yükleme hatası: {e}")
        exit()
    
    model.eval()
    class_names = ['Normal', 'Mild', 'Moderate/Severe']
    
    # Validation
    print("\n🚀 Validation Evaluation...")
    val_acc, val_bal, val_aucs = evaluate_set(
        model, val_loader, device,
        set_name="VALIDATION",
        class_names=class_names,
        save_prefix="fusion_val"
    )
    
    # Test
    print("\n🚀 Test Evaluation...")
    test_acc, test_bal, test_aucs = evaluate_set(
        model, test_loader, device,
        set_name="TEST",
        class_names=class_names,
        save_prefix="fusion_test"
    )
    
    # Özet
    print("\n" + "="*60)
    print("GENERAL SUMMARY")
    print("="*60)
    print(f"{'Metric':<25} {'Validation':>12} {'Test':>12}")
    print("-"*50)
    print(f"{'Accuracy':<25} {val_acc*100:>11.2f}% {test_acc*100:>11.2f}%")
    print(f"{'Balanced Accuracy':<25} {val_bal*100:>11.2f}% {test_bal*100:>11.2f}%")
    for name in class_names:
        print(f"{'AUC ' + name:<25} {val_aucs[name]:>12.3f} {test_aucs[name]:>12.3f}")
    
    plt.show()
    print("\n✅ Tamamlandı!")

    # ============================================================
    # FEATURE IMPORTANCE ANALIZI (Permutation Importance)
    # ============================================================
    print("\n" + "="*60)
    print("🔍 FEATURE IMPORTANCE ANALYSIS (Permutation Importance)")
    print("="*60)
    
    def analyze_feature_importance(model, loader, device, feature_names):
        model.eval()
        
        # 1. Tüm veriyi CPU'da topla (GPU belleğini şişirmemek için)
        all_mri = []
        all_clinical = []
        all_labels = []
        
        print("Gathering data (CPU)...")
        with torch.no_grad():
            for mri, clinical, labels in loader:
                all_mri.append(mri.cpu())
                all_clinical.append(clinical.cpu())
                all_labels.append(labels.cpu())
        
        all_mri = torch.cat(all_mri)
        all_clinical = torch.cat(all_clinical)
        all_labels = torch.cat(all_labels)
        
        print(f"Total data shape: MRI {all_mri.shape}, Clinical {all_clinical.shape}")

        criterion = nn.CrossEntropyLoss()
        
        # Batch işlemleri için yardımcı fonksiyon
        def compute_loss_batched(mri_data, clinical_data, labels_data, batch_size=4):
            total_loss = 0.0
            total_count = 0
            dataset_size = len(labels_data)
            
            # Basit bir döngü ile batch batch işle
            with torch.no_grad():
                for i in range(0, dataset_size, batch_size):
                    end_idx = min(i + batch_size, dataset_size)
                    
                    batch_mri = mri_data[i:end_idx].to(device)
                    batch_clinical = clinical_data[i:end_idx].to(device)
                    batch_labels = labels_data[i:end_idx].to(device)
                    
                    outputs = model(batch_mri, batch_clinical)
                    loss = criterion(outputs, batch_labels)
                    
                    # Batch ağırlıklı ortalama için toplam kaybı biriktir
                    batch_count = end_idx - i
                    total_loss += loss.item() * batch_count
                    total_count += batch_count
            
            return total_loss / total_count if total_count > 0 else 0.0

        # 2. Baseline Performans
        print("Calculating baseline loss...")
        baseline_loss = compute_loss_batched(all_mri, all_clinical, all_labels)
        print(f"Baseline Test Loss: {baseline_loss:.4f}\n")
        
        feature_importance = {}
        
        # 3. Her özellik için Permutation
        print("Analyzing features (this might take a while)...")
        # tqdm progress bar
        for i, feature_name in enumerate(tqdm(feature_names)):
            # Orijinal kolonu sakla (CPU üzerinde)
            original_col = all_clinical[:, i].clone()
            
            # Kolonu karıştır (Shuffle) - CPU üzerinde
            permuted_col = all_clinical[:, i][torch.randperm(len(all_clinical))]
            all_clinical[:, i] = permuted_col
            
            # Batched loss hesapla
            loss = compute_loss_batched(all_mri, all_clinical, all_labels)
            
            # Loss artışı = Önem derecesi
            importance = loss - baseline_loss
            feature_importance[feature_name] = importance
            
            # Orijinal kolonu geri yükle
            all_clinical[:, i] = original_col
            
        return feature_importance

    # Analizi Çalıştır
    importances = analyze_feature_importance(model, test_loader, device, clinical_cols)
    
    # Sırala ve Yazdır (En önemli 20)
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    print("\n🏆 TOP 20 MOST IMPORTANT CLINICAL FEATURES (By Loss Increase):")
    print("-" * 60)
    print(f"{'Rank':<5} {'Feature Name':<35} {'Importance Score (Loss Diff)':<20}")
    print("-" * 60)
    
    for i, (name, score) in enumerate(sorted_importances[:20]):
        print(f"{i+1:<5} {name:<35} {score:.6f}")
        
    print("\n📉 LEAST IMPORTANT 20 CLINICAL FEATURES:")
    print("-" * 60)
    print(f"{'Rank':<5} {'Feature Name':<35} {'Importance Score (Loss Diff)':<20}")
    print("-" * 60)
    
    # En kötüden iyiye doğru (listenin sonu)
    for i, (name, score) in enumerate(sorted_importances[-20:]):
        print(f"{len(sorted_importances)-20+i+1:<5} {name:<35} {score:.6f}")

    # CSV Kaydet
    imp_df = pd.DataFrame(sorted_importances, columns=['Feature', 'Importance'])
    imp_df.to_csv("feature_importance_all.csv", index=False)
    print(f"\n✅ Tüm özellik skorları kaydedildi: feature_importance_all.csv")

    # Görselleştirme
    plt.figure(figsize=(12, 10))
    names = [x[0] for x in sorted_importances[:20]]
    scores = [x[1] for x in sorted_importances[:20]]
    
    sns.barplot(x=scores, y=names, palette="viridis")
    plt.title("Top 20 Most Important Clinical Features for Model (Permutation Importance)")
    plt.xlabel("Loss Increase (Higher = More Important)")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("\n✅ Grafik kaydedildi: feature_importance.png")
    plt.show()
