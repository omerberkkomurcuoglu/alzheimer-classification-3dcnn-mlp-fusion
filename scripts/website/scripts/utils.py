"""
Alzheimer Tahmin Sistemi - Yardımcı Fonksiyonlar
predict.ipynb ve app.py tarafından kullanılır

Model: CNN_MLP_Fusion (Custom3DCNN + MLP with Cross-Modal Attention)
Kaynak: 15.0.1-CustomCnn_MLP_Fusion.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import zoom
import nibabel as nib
import os

# ============================================
# SABİTLER
# ============================================
CLASS_NAMES = {0: "CN (Normal)", 1: "MCI (Hafif Bilişsel Bozukluk)", 2: "AD (Alzheimer)"}
NUM_CLASSES = 3
CLINICAL_DIM = 72  # 12-data_leakage.csv'den gelen klinik sütun sayısı


# ============================================
# MODEL MİMARİSİ - Custom3DCNN Bileşenleri
# (custom_model.py'den)
# ============================================
class ResidualBlock(nn.Module):
    """3D Residual Blok (GroupNorm ile)"""
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, bias=False)
        self.gn1 = nn.GroupNorm(min(8, out_ch), out_ch)
        
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(8, out_ch), out_ch)
        
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout3d(dropout)
        
        # Boyut eşitleme
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.gn2(self.conv2(out))
        
        out += residual
        out = self.relu(out)
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1, 1)
        return x * y


class Custom3DCNN(nn.Module):
    """
    Custom 3D CNN - MRI Sınıflandırması
    
    Mimari:
    - Input Normalization (InstanceNorm3d)
    - Giriş: (B, 1, 128, 128, 128)
    - 4 Residual blok (32->64->128->256)
    - SE blokları (son 2 katmanda)
    - Global Average Pooling
    - FC sınıflandırıcı
    - Çıkış: (B, num_classes)
    """
    def __init__(self, num_classes=3, dropout=0.2):
        super(Custom3DCNN, self).__init__()
        
        self.input_norm = nn.InstanceNorm3d(1, affine=True)
        
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=False)
        )
        
        self.layer1 = ResidualBlock(32, 64, stride=2, dropout=0.05)
        
        self.layer2 = ResidualBlock(64, 128, stride=2, dropout=0.1)
        self.se2 = SEBlock(128, reduction=8)
        
        self.layer3 = ResidualBlock(128, 256, stride=2, dropout=0.15)
        self.se3 = SEBlock(256, reduction=16)
        
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        last_linear = self.classifier[-1]
        nn.init.normal_(last_linear.weight, mean=0, std=0.01)
        nn.init.zeros_(last_linear.bias)
    
    def forward(self, x):
        x = self.input_norm(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        x = self.se3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


# ============================================
# FUSION MODEL (15.0.1-CustomCnn_MLP_Fusion.py)
# ============================================
class CNN_MLP_Fusion(nn.Module):
    def __init__(self, clinical_dim=72, num_classes=3, dropout=0.3,
                 pretrained_cnn_path=None, pretrained_mlp_path=None,
                 modality_dropout_rate=0.4):
        super().__init__()

        self.modality_dropout_rate = modality_dropout_rate

        # === CNN Branch ===
        cnn_full = Custom3DCNN(num_classes=num_classes, dropout=dropout)

        if pretrained_cnn_path and os.path.exists(pretrained_cnn_path):
            state_dict = torch.load(pretrained_cnn_path, map_location='cpu')
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            cnn_full.load_state_dict(state_dict, strict=True)

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

            self.load_state_dict(new_state_dict, strict=False)

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

    def forward(self, mri, clinical, current_modality_dropout=None):
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

        # Asymmetric Modality Dropout (scheduled rate)
        drop_rate = current_modality_dropout if current_modality_dropout is not None else self.modality_dropout_rate
        if self.training and drop_rate > 0:
            r = torch.rand(1).item()
            if r < drop_rate:
                if torch.rand(1).item() < 0.7:  # 70% clinical drop
                    mlp_features = torch.zeros_like(mlp_features)
                else:  # 30% MRI drop
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


# ============================================
# MODEL YÜKLEME
# ============================================
def load_model(model_path, device=None):
    """Eğitilmiş CNN_MLP_Fusion modelini yükle"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CNN_MLP_Fusion(
        clinical_dim=CLINICAL_DIM,
        num_classes=NUM_CLASSES,
        dropout=0.3,
        pretrained_cnn_path=None,
        pretrained_mlp_path=None,
        modality_dropout_rate=0.4
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


# ============================================
# EMBEDDING FONKSİYONLARI
# ============================================
def load_embedding_dataframes(meds_path, dx_path):
    """Embedding CSV dosyalarını yükle"""
    meds_df = pd.read_csv(meds_path)
    dx_df = pd.read_csv(dx_path)
    
    meds_df['medicine_lower'] = meds_df['medicine'].str.lower().str.strip()
    dx_df['dx_lower'] = dx_df['dx'].str.lower().str.strip()
    
    return meds_df, dx_df


def get_medicine_embedding(medicine_name, meds_df):
    """İlaç adından embedding değerlerini çek"""
    if not medicine_name or pd.isna(medicine_name) or medicine_name == "":
        return {f'emb_{i}': 0.0 for i in range(16)}
    
    medicine_lower = str(medicine_name).lower().strip()
    
    # Tam eşleşme ara
    match = meds_df[meds_df['medicine_lower'] == medicine_lower]
    
    # Bulunamazsa kısmi eşleşme dene
    if match.empty:
        match = meds_df[meds_df['medicine_lower'].str.contains(medicine_lower, na=False)]
    
    if not match.empty:
        row = match.iloc[0]
        return {f'emb_{i}': float(row[str(i)]) for i in range(16)}
    else:
        print(f"⚠️ İlaç bulunamadı: '{medicine_name}' - Varsayılan 0 kullanılıyor")
        return {f'emb_{i}': 0.0 for i in range(16)}


def get_diagnosis_embedding(dx_name, dx_df):
    """Tanı adından embedding değerlerini çek"""
    if not dx_name or pd.isna(dx_name) or dx_name == "":
        return {f'emb_{i}': 0.0 for i in range(4)}
    
    dx_lower = str(dx_name).lower().strip()
    
    # Tam eşleşme ara
    match = dx_df[dx_df['dx_lower'] == dx_lower]
    
    # Bulunamazsa kısmi eşleşme dene
    if match.empty:
        match = dx_df[dx_df['dx_lower'].str.contains(dx_lower, na=False)]
    
    if not match.empty:
        row = match.iloc[0]
        return {f'emb_{i}': float(row[str(i)]) for i in range(4)}
    else:
        print(f"⚠️ Tanı bulunamadı: '{dx_name}' - Varsayılan 0 kullanılıyor")
        return {f'emb_{i}': 0.0 for i in range(4)}


def build_embeddings_from_names(meds_df, dx_df, 
                                 med1=None, med2=None, med3=None, med4=None,
                                 dx1=None, dx2=None, dx3=None, dx4=None, dx5=None):
    """İlaç ve tanı isimlerinden tüm embedding değerlerini oluştur"""
    embeddings = {}
    
    # İlaç embeddingleri (her biri 16 boyut)
    meds = [med1, med2, med3, med4]
    for i, med in enumerate(meds, 1):
        med_emb = get_medicine_embedding(med, meds_df)
        for j in range(16):
            embeddings[f'meds_{i}_emb_{j}'] = med_emb[f'emb_{j}']
    
    # Tanı embeddingleri (her biri 4 boyut)
    dxs = [dx1, dx2, dx3, dx4, dx5]
    for i, dx in enumerate(dxs, 1):
        dx_emb = get_diagnosis_embedding(dx, dx_df)
        for j in range(4):
            embeddings[f'dx{i}_emb_{j}'] = dx_emb[f'emb_{j}']
    
    return embeddings


# ============================================
# SCALER VE KLİNİK SÜTUNLAR
# ============================================
def create_scaler_from_training_data(data_csv_path):
    """StandardScaler'ı eğitim verisinden oluştur"""
    train_data = pd.read_csv(data_csv_path)
    
    # Klinik sütunları belirle (OASISID, memory, path hariç)
    clinical_cols = [col for col in train_data.columns 
                     if col not in ['OASISID', 'memory', 'path']]
    
    # Temizlik
    for col in clinical_cols:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce').fillna(0)
    
    # Scaler'ı fit et
    scaler = StandardScaler()
    scaler.fit(train_data[clinical_cols])
    
    # Ortalama değerler (varsayılan değerler için)
    mean_dict = train_data[clinical_cols].mean().to_dict()
    
    return scaler, clinical_cols, mean_dict


# ============================================
# MR GÖRÜNTÜSÜ İŞLEME
# ============================================
def load_and_preprocess_mr(file_path, target_shape=(128, 128, 128)):
    """
    MR görüntüsünü yükler ve modele uygun formata getirir.
    - .npy: Zaten 0-1 normalize edilmiş, sadece Z-score uygula
    - .nii/.nii.gz: Önce 0-1 normalize, resize, sonra Z-score uygula
    """
    file_ext = file_path.lower()
    
    if file_ext.endswith('.npy'):
        data = np.load(file_path).astype(np.float32)
        
    elif file_ext.endswith('.nii') or file_ext.endswith('.nii.gz'):
        nii_img = nib.load(file_path)
        data = nii_img.get_fdata().astype(np.float32)
        
        # Min-Max Normalizasyon (0-1 arası)
        data_min, data_max = np.min(data), np.max(data)
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)
        
        # Resize (128x128x128'e)
        if data.shape != target_shape:
            zoom_factors = [t / s for t, s in zip(target_shape, data.shape)]
            data = zoom(data, zoom_factors, order=1)
    else:
        raise ValueError(f"Desteklenmeyen dosya formatı: {file_path}")
    
    # Boyut kontrolü
    if data.shape != target_shape:
        zoom_factors = [t / s for t, s in zip(target_shape, data.shape)]
        data = zoom(data, zoom_factors, order=1)
    
    # Z-score normalizasyonu
    mean = data.mean()
    std = data.std()
    if std > 0:
        data = (data - mean) / (std + 1e-6)
    else:
        data = data - mean
    
    return data


def load_and_preprocess_mr_from_bytes(file_data, file_name, target_shape=(128, 128, 128)):
    """
    Byte stream'den MR görüntüsünü yükler (Streamlit file_uploader için)
    """
    import io
    import tempfile
    
    file_ext = file_name.lower()
    
    if file_ext.endswith('.npy'):
        data = np.load(io.BytesIO(file_data)).astype(np.float32)
        
    elif file_ext.endswith('.nii') or file_ext.endswith('.nii.gz'):
        suffix = '.nii.gz' if file_ext.endswith('.nii.gz') else '.nii'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name
        
        nii_img = nib.load(tmp_path)
        data = nii_img.get_fdata().astype(np.float32)
        os.unlink(tmp_path)
        
        # Min-Max Normalizasyon
        data_min, data_max = np.min(data), np.max(data)
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)
        
        # Resize
        if data.shape != target_shape:
            zoom_factors = [t / s for t, s in zip(target_shape, data.shape)]
            data = zoom(data, zoom_factors, order=1)
    else:
        raise ValueError(f"Desteklenmeyen dosya formatı: {file_name}")
    
    # Boyut kontrolü
    if data.shape != target_shape:
        zoom_factors = [t / s for t, s in zip(target_shape, data.shape)]
        data = zoom(data, zoom_factors, order=1)
    
    # Z-score normalizasyonu
    mean = data.mean()
    std = data.std()
    if std > 0:
        data = (data - mean) / (std + 1e-6)
    else:
        data = data - mean
    
    return data


# ============================================
# KLİNİK VERİ İŞLEME
# ============================================
def prepare_clinical_data(clinical_dict, clinical_cols, scaler):
    """Klinik verileri modele uygun formata çevirir ve normalize eder"""
    clinical_values = []
    for col in clinical_cols:
        val = clinical_dict.get(col, 0)
        try:
            val = float(val)
        except:
            val = 0.0
        # NaN kontrolü
        if np.isnan(val):
            val = 0.0
        clinical_values.append(val)
    
    # NumPy array'e çevir
    clinical_array = np.array(clinical_values, dtype=np.float32).reshape(1, -1)
    
    # StandardScaler ile normalize et
    clinical_normalized = scaler.transform(clinical_array)
    
    return clinical_normalized.flatten().astype(np.float32)


# ============================================
# TAHMİN FONKSİYONU
# ============================================
def predict_alzheimer(mr_data, clinical_dict, model, device, clinical_cols, scaler):
    """MR görüntüsü ve klinik verilerden Alzheimer tahmini yapar"""
    model.eval()
    
    # 1. MR tensörü hazırla
    mr_tensor = torch.tensor(mr_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mr_tensor = mr_tensor.to(device)
    
    # 2. Klinik verileri hazırla ve normalize et
    clinical_data = prepare_clinical_data(clinical_dict, clinical_cols, scaler)
    clinical_tensor = torch.tensor(clinical_data, dtype=torch.float32).unsqueeze(0)
    clinical_tensor = clinical_tensor.to(device)
    
    # 3. Tahmin yap
    with torch.no_grad():
        outputs = model(mr_tensor, clinical_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        prediction = torch.argmax(outputs, dim=1).item()
    
    class_name = CLASS_NAMES[prediction]
    
    return prediction, probabilities, class_name


# ============================================
# JSON/CSV HASTA VERİSİ OKUMA
# ============================================
def load_patient_from_json(json_path):
    """JSON dosyasından hasta verisini yükle"""
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_patient_from_csv(csv_path):
    """CSV dosyasından hasta verisini yükle"""
    df = pd.read_csv(csv_path)
    row = df.iloc[0]
    
    medicines = {
        'med1': row.get('med1', None),
        'med2': row.get('med2', None),
        'med3': row.get('med3', None),
        'med4': row.get('med4', None),
    }
    
    diagnoses = {
        'dx1': row.get('dx1', None),
        'dx2': row.get('dx2', None),
        'dx3': row.get('dx3', None),
        'dx4': row.get('dx4', None),
        'dx5': row.get('dx5', None),
    }
    
    clinical_data = {col: row[col] for col in row.index 
                     if col not in ['med1','med2','med3','med4','dx1','dx2','dx3','dx4','dx5','mr_file_path']}
    
    return {
        'mr_file_path': row.get('mr_file_path', ''),
        'medicines': medicines,
        'diagnoses': diagnoses,
        'clinical_data': clinical_data
    }
