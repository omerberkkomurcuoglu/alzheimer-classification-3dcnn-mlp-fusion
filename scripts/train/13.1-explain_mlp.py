
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import os

# ============================================================
# 1. SEED & AYARLAR
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed sabitlendi: {seed}")

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. DATASET & MODEL (Training scripti ile aynı olmalı)
# ============================================================
class ClinicalOnlyDataset(Dataset):
    def __init__(self, data_csv, augment=False):
        self.data_csv = data_csv.reset_index(drop=True)
        self.augment = augment
        self.label_map = {0: 0, 0.5: 1, 1: 2, 2: 2}
        self.clinical_cols = [col for col in self.data_csv.columns if col not in ['OASISID', 'memory', 'path']]

    def __len__(self):
        return len(self.data_csv)
    
    def __getitem__(self, idx):
        row = self.data_csv.iloc[idx]
        label_val = row['memory']
        try:
            label_val = float(label_val)
        except:
            label_val = 0.0
        label = self.label_map.get(label_val, 0)
        
        clinical_series = row[self.clinical_cols]
        clinical_data = pd.to_numeric(clinical_series, errors='coerce').fillna(0).values.astype(np.float32)
        
        return torch.tensor(clinical_data), torch.tensor(label, dtype=torch.long)

class MLP_Only(nn.Module):
    def __init__(self, input_dim=170, num_classes=3):
        super(MLP_Only, self).__init__()
        # Dropout oranlarını training ile aynı tutuyoruz (0.3, 0.3, 0.2)
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.residual_proj = nn.Linear(256, 64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        residual = self.residual_proj(x1)
        x3 = x3 + residual
        out = self.classifier(x3)
        return out

def analyze_feature_importance(model, loader, device, feature_names):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # 1. Tüm veriyi topla
    print("Veriler toplanıyor...")
    all_clinical = []
    all_labels = []
    for clinical, labels in loader:
        all_clinical.append(clinical)
        all_labels.append(labels)
        
    all_clinical = torch.cat(all_clinical).to(device)
    all_labels = torch.cat(all_labels).to(device).long()
    
    # 2. Baseline Loss hesapla
    print("Baseline Loss hesaplanıyor...")
    with torch.no_grad():
        outputs = model(all_clinical)
        baseline_loss = criterion(outputs, all_labels).item()
    
    print(f"Baseline Loss: {baseline_loss:.6f}")
    
    feature_importance = []
    
    # 3. Her özellik için Permutation Importance
    for i, feature_name in enumerate(tqdm(feature_names, desc="Analyzing Features")):
        shuffled_clinical = all_clinical.clone()
        
        # Karıştır
        perm = torch.randperm(shuffled_clinical.size(0))
        shuffled_clinical[:, i] = shuffled_clinical[perm, i]
        
        with torch.no_grad():
            outputs = model(shuffled_clinical)
            loss = criterion(outputs, all_labels).item()
        
        importance = loss - baseline_loss
        feature_importance.append({
            'Feature': feature_name,
            'Importance': importance
        })
        
    return pd.DataFrame(feature_importance).sort_values(by='Importance', ascending=False)

# ============================================================
# 3. MAIN
# ============================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "12-data_leakage.csv")
    
    if not os.path.exists(data_path):
        print(f"HATA: Veri dosyası bulunamadı: {data_path}")
        exit(1)
        
    data_csv = pd.read_csv(data_path)
    clinical_cols = [col for col in data_csv.columns if col not in ['OASISID', 'memory', 'path']]
    
    # Veri Hazırlığı (Training script ile aynı mantıkta scale edilmeli)
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_inds, test_inds = next(splitter_test.split(data_csv, groups=data_csv['OASISID']))

    train_val_df = data_csv.iloc[train_val_inds].reset_index(drop=True)
    test_df = data_csv.iloc[test_inds].reset_index(drop=True)

    val_ratio = 15 / 85
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_inds, val_inds = next(splitter_val.split(train_val_df, groups=train_val_df['OASISID']))

    train_df = train_val_df.iloc[train_inds].reset_index(drop=True)
    val_df = train_val_df.iloc[val_inds].reset_index(drop=True)

    # Scale
    for col in clinical_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        val_df[col] = pd.to_numeric(val_df[col], errors='coerce').fillna(0)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
    
    scaler = StandardScaler()
    scaler.fit(train_df[clinical_cols]) # Train set üzerinden fit
    test_df[clinical_cols] = scaler.transform(test_df[clinical_cols])
    
    test_dataset = ClinicalOnlyDataset(test_df, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    print(f"Test seti boyutu: {len(test_dataset)}")
    
    # Model Yükle
    model = MLP_Only(input_dim=len(clinical_cols), num_classes=3)
    
    weight_path = "mlp_only_best_bal_acc.pth"
    if not os.path.exists(weight_path):
        print(f"UYARI: {weight_path} bulunamadı, 'mlp_only_best.pth' aranıyor...")
        weight_path = "mlp_only_best.pth"
    
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        print(f"Model yüklendi: {weight_path}")
        
        # Analiz Başlat
        df_importance = analyze_feature_importance(model, test_loader, device, clinical_cols)
        
        # Kaydet
        df_importance.to_csv("mlp_feature_importance_fast.csv", index=False)
        print("\n✅ Sonuçlar kaydedildi: mlp_feature_importance_fast.csv")
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=df_importance.head(30), palette='viridis')
        plt.title(f'Top 30 Feature Importance (MLP) - {weight_path}')
        plt.xlabel('Loss Increase (Importance)')
        plt.tight_layout()
        plt.savefig("mlp_feature_importance_fast.png", dpi=150)
        plt.show()
        print("✅ Grafik kaydedildi: mlp_feature_importance_fast.png")
        
    else:
        print("HATA: Kayıtlı model ağırlığı bulunamadı! Lütfen önce eğitimi tamamlayın.")
