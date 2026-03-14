"""
15.2 - Modality Importance Analysis script
Fusion Model'in MRI ve Klinik veriye ne kadar bağımlı olduğunu test eder.
Yöntem: 
1. Baseline Accuracy (Orijinal)
2. MRI Shuffled Accuracy (MRI verisi karıştırılmış) -> Düşüş = MRI Önemi
3. Clinical Shuffled Accuracy (Klinik veri karıştırılmış) -> Düşüş = Klinik Önemi
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
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
from custom_model import Custom3DCNN

class CNN_MLP_Fusion(nn.Module):
    """Custom3DCNN + MLP Fusion (v3 - Cross-Modal Attention)"""
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
        
        self.cnn_aux = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        # === MLP Branch ===
        self.mlp_block1 = nn.Sequential(
            nn.Linear(clinical_dim, 256), nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True), nn.Dropout(dropout),
        )
        self.mlp_block2 = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True), nn.Dropout(dropout),
        )
        self.mlp_block3 = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True), nn.Dropout(dropout * 0.5),
        )
        self.mlp_residual_proj = nn.Linear(256, 64)
        
        self.mlp_aux = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
        # === Feature Projection ===
        self.cnn_proj = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU()
        )
        self.mlp_proj = nn.Sequential(
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU()
        )
        
        # === Cross-Modal Attention Gates ===
        total_dim = 256
        self.cnn_gate = nn.Sequential(nn.Linear(total_dim, 128), nn.Sigmoid())
        self.mlp_gate = nn.Sequential(nn.Linear(total_dim, 128), nn.Sigmoid())
        
        # === Classifier ===
        self.fusion_classifier = nn.Sequential(
            nn.Linear(total_dim, 128), nn.BatchNorm1d(128),
            nn.ReLU(inplace=False), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.ReLU(inplace=False), nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, mri, clinical):
        import torch.nn.functional as F
        
        x = self.input_norm(mri)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        x = self.se3(x)
        x = self.gap(x)
        cnn_features = x.view(x.size(0), -1)
        
        m1 = self.mlp_block1(clinical)
        m2 = self.mlp_block2(m1)
        m3 = self.mlp_block3(m2)
        mlp_features = m3 + self.mlp_residual_proj(m1)
        
        cnn_proj = F.normalize(self.cnn_proj(cnn_features), p=2, dim=1)
        mlp_proj = F.normalize(self.mlp_proj(mlp_features), p=2, dim=1)
        
        combined = torch.cat([cnn_proj, mlp_proj], dim=1)
        gated_cnn = cnn_proj * self.cnn_gate(combined)
        gated_mlp = mlp_proj * self.mlp_gate(combined)
        gated = torch.cat([gated_cnn, gated_mlp], dim=1)
        
        return self.fusion_classifier(gated)

# ============================================================
# 3. ANALİZ
# ============================================================
def evaluate_modality_importance(model, loader, device):
    model.eval()
    
    # Tüm veriyi belleğe al (Daha hızlı shuffle için)
    print("Veriler belleğe yükleniyor...")
    all_mri = []
    all_clinical = []
    all_labels = []
    
    with torch.no_grad():
        for mri, clinical, labels in tqdm(loader, desc="Loading Data"):
            all_mri.append(mri)
            all_clinical.append(clinical)
            all_labels.append(labels)
    
    all_mri = torch.cat(all_mri).to(device)
    all_clinical = torch.cat(all_clinical).to(device)
    all_labels = torch.cat(all_labels).to(device)
    
    print(f"Veri yüklendi: {len(all_labels)} örnek")
    
    def get_accuracy(mri_data, clinical_data, labels_data):
        """Batch batch tahmin alıp accuracy hesaplar"""
        preds = []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(labels_data), batch_size):
                batch_mri = mri_data[i:i+batch_size]
                batch_clinical = clinical_data[i:i+batch_size]
                
                outputs = model(batch_mri, batch_clinical)
                batch_preds = torch.argmax(outputs, dim=1)
                preds.append(batch_preds)
        
        preds = torch.cat(preds)
        acc = (preds == labels_data).float().mean().item()
        return acc

    # 1. Baseline
    print("\n--- Baseline Hesapla ---")
    baseline_acc = get_accuracy(all_mri, all_clinical, all_labels)
    print(f"Baseline Accuracy: {baseline_acc*100:.2f}%")
    
    # 2. Shuffle MRI (MRI Önemi)
    print("\n--- MRI Etkisini Ölç (Shuffling MRI) ---")
    # MRI verisini karıştır (Clinical ve Label sabit)
    indices = torch.randperm(len(all_mri))
    shuffled_mri = all_mri[indices]
    
    mri_shuffled_acc = get_accuracy(shuffled_mri, all_clinical, all_labels)
    mri_drop = baseline_acc - mri_shuffled_acc
    print(f"MRI Shuffled Accuracy: {mri_shuffled_acc*100:.2f}%")
    print(f"DROP (MRI Importance): {mri_drop*100:.2f} puan")
    
    # 3. Shuffle Clinical (Klinik Önemi)
    print("\n--- Klinik Etkisini Ölç (Shuffling Clinical) ---")
    # Clinical verisini karıştır (MRI ve Label sabit)
    indices = torch.randperm(len(all_clinical))
    shuffled_clinical = all_clinical[indices]
    
    clinical_shuffled_acc = get_accuracy(all_mri, shuffled_clinical, all_labels)
    clinical_drop = baseline_acc - clinical_shuffled_acc
    print(f"Clinical Shuffled Accuracy: {clinical_shuffled_acc*100:.2f}%")
    print(f"DROP (Clinical Importance): {clinical_drop*100:.2f} puan")
    
    return baseline_acc, mri_shuffled_acc, clinical_shuffled_acc

# ============================================================
# 4. MAIN
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Dosya yolları
    processed_dir = "C:\\Users\\berkk\\Desktop\\Bitirme_Projesi\\processed_dataset_128"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_csv = pd.read_csv(os.path.join(script_dir, "12-data_leakage.csv"))
    
    clinical_cols = [col for col in data_csv.columns if col not in ['OASISID', 'memory', 'path']]
    CLINICAL_DIM = len(clinical_cols)
    print(f"Toplam veri: {len(data_csv)} | Klinik özellik: {CLINICAL_DIM}")
    
    # Split (Aynı split)
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    _, test_inds = next(splitter_test.split(data_csv, groups=data_csv['OASISID']))
    test_df = data_csv.iloc[test_inds].reset_index(drop=True)
    
    # Normalizasyon
    scaler = StandardScaler()
    scaler.fit(data_csv[clinical_cols])
    test_df[clinical_cols] = scaler.transform(test_df[clinical_cols])
    
    test_dataset = MRIClinicalDataset(processed_dir, test_df, clinical_cols)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Model
    model = CNN_MLP_Fusion(clinical_dim=CLINICAL_DIM, num_classes=3, dropout=0.3)
    model.to(device)
    
    weights_path = os.path.join(script_dir, "weights", "Fusion_Full_best_balanced.pth")
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"\n✅ Model yüklendi: {os.path.basename(weights_path)}")
    except Exception as e:
        print(f"\n❌ Model yükleme hatası: {e}")
        exit()
        
    # Analiz Başlat
    baseline, mri_acc, clin_acc = evaluate_modality_importance(model, test_loader, device)
    
    # Grafik Çiz
    labels = ['Baseline', 'MRI Shuffled', 'Clinical Shuffled']
    accuracies = [baseline*100, mri_acc*100, clin_acc*100]
    colors = ['gray', 'blue', 'orange']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, accuracies, color=colors)
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.title('Modality Importance Analysis (Test Set)')
    
    # Değerleri yaz
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', fontweight='bold')
        
    # Drop oranlarını ok ile göster
    # MRI Drop
    plt.annotate(f"Drop: -{(baseline-mri_acc)*100:.1f}%", 
                 xy=(1, mri_acc*100), xytext=(1, baseline*100),
                 arrowprops=dict(facecolor='red', arrowstyle='->'), ha='center')
    
    # Clinical Drop
    plt.annotate(f"Drop: -{(baseline-clin_acc)*100:.1f}%", 
                 xy=(2, clin_acc*100), xytext=(2, baseline*100),
                 arrowprops=dict(facecolor='red', arrowstyle='->'), ha='center')
    
    plt.tight_layout()
    plt.savefig("modality_importance.png", dpi=150)
    print("\n✅ Grafik kaydedildi: modality_importance1.png")
    
    # Yorum
    print("\n" + "="*50)
    print("SONUÇ YORUMU")
    print("="*50)
    if (baseline - mri_acc) > 0.05:
        print("✅ MRI Görüntüleri model için KRİTİK öneme sahip.")
        print(f"   MRI bozulduğunda doğruluk {baseline*100:.1f}% -> {mri_acc*100:.1f}% düşüyor.")
    else:
        print("⚠️ MRI Görüntülerinin etkisi sınırlı görünüyor.")
        print(f"   Düşüş sadece {(baseline-mri_acc)*100:.1f}% puan.")
        
    if (baseline - clin_acc) > 0.10:
        print("✅ Klinik Veriler modelin ana taşıyıcısı.")
