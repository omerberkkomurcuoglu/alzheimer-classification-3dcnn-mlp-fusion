import pandas as pd
import os

# Dosya yolları
script_dir = os.path.dirname(os.path.abspath(__file__))
# 11-data.csv, tüm projenin ham birleştirilmiş kaynak verisidir.
input_file = os.path.join(script_dir, "11-data.csv")
# Optimize edilmiş yeni çıktı (MLP için en iyi sızıntı temizlenmiş hali)
output_file = os.path.join(script_dir, "12-data_leakage_optimized.csv")

# Veriyi yükle
print(f"Loading {input_file}...")
df = pd.read_csv(input_file)

# Çıkarılacak GERÇEK sızıntı (leakage) sütunları
# Bu sütunlar hastalığın teşhisinde doğrudan kullanıldığı için kopya çekmektir.
leakage_cols = [
    # CDR Skorları (Teşhisin kendisi)
    "CDRSUM", "CDRTOT",
    
    # CDR Bileşenleri (Bu puanların toplamı CDR'ı verir)
    "orient", "judgment", "commun", "homehobb", "perscare",
    
    # Bilişsel Testler (Alzheimer teşhisinde ana kriterler)
    "MMSE",
    
    # Doğrudan bilişsel durum kararı içerenler (Ground Truth kopyaları)
    "DECSUB", "DECIN", "NORMCOG", "MEMPROB"
]

# dx sütunları da çıkarılmalı (doğrudan teşhis bilgisi ve eski kodları içeriyor)
dx_leakage_cols = [col for col in df.columns if col.startswith('dx')]
print(f"Found {len(dx_leakage_cols)} dx columns (will be dropped): {dx_leakage_cols}")

# Negatif etkili veya sıfır etkili klinik sütunlar + İlaç (Medication) sütunları 
# ARTIK SİLİNMİYOR! MLP'nin özellik çıkarımı (Non-Linear Feature Extraction) yapması sağlanıyor.

# Tüm çıkarılacak sütunları birleştir
all_leakage_cols = leakage_cols + dx_leakage_cols

# Mevcut olanları bul ve çıkar
cols_to_drop = [col for col in all_leakage_cols if col in df.columns]
print(f"Dropping {len(cols_to_drop)} real leakage columns: {cols_to_drop}")

df_cleaned = df.drop(columns=cols_to_drop)

# Kaydet
df_cleaned.to_csv(output_file, index=False)
print(f"Cleaned (Optimized) dataset saved to {output_file}")
print(f"Original shape: {df.shape}")
print(f"New shape: {df_cleaned.shape}")
