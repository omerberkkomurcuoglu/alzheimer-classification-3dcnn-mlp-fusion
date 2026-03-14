import pandas as pd
import os

# Dosya yolları
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "11-data.csv")
output_file = os.path.join(script_dir, "12-data_leakage_non_meds.csv")

# Veriyi yükle
print(f"Loading {input_file}...")
df = pd.read_csv(input_file)

# Çıkarılacak sızıntı (leakage) sütunları
# Bu sütunlar hastalığın teşhisinde doğrudan kullanıldığı için 'cevap anahtarı' niteliğindedir.
leakage_cols = [
    # CDR Skorları (Teşhisin kendisi)
    "CDRSUM", "CDRTOT",
    
    # CDR Bileşenleri (Bu puanların toplamı CDR'ı verir)
    "orient", "judgment", "commun", "homehobb", "perscare",
    
    # Bilişsel Testler (Alzheimer teşhisinde ana kriterler)
    "MMSE",
    
    # 0 Etkili Özellikler (Feature Importance)
    "PSP", "CORT", "HUNT", "PRION", "DOWNS", "NEOP", "COGOTH3",

    # Negatif Etkili Özellikler (Feature Importance - Loss azaltanlar)
    "INSEX", "HOPELESS", "COGOTH2", "DEL", "HELPLESS", "AFRAID", "STAYHOME",
    "STOVE", "WRTHLESS", "DISN", "WEIGHT", "DEPD", "TRAVEL", "ENERGY",
    "EVENTS", "NITE", "PARK", "DROPACT", "SHOPPING", "MOT", "BRNINJ",
    "EMPTY", "WONDRFUL", "MEALPREP", "BETTER", "SATIS",
    
    # Doğrudan bilişsel durum kararı içerenler
    "DECSUB", "DECIN", "NORMCOG", "MEMPROB"
]

# dx sütunları da çıkarılmalı (doğrudan teşhis bilgisi içeriyor)
dx_leakage_cols = [col for col in df.columns if col.startswith('dx')]
print(f"Found {len(dx_leakage_cols)} dx columns: {dx_leakage_cols}")

# İlaç sütunları (embedding'ler dahil)
med_cols = [col for col in df.columns if 'med' in col.lower() or col == 'MEDS']
print(f"Found {len(med_cols)} medication columns: {med_cols}")

# Tüm çıkarılacak sütunları birleştir
all_leakage_cols = leakage_cols + dx_leakage_cols + med_cols

# Mevcut olanları bul ve çıkar
cols_to_drop = [col for col in all_leakage_cols if col in df.columns]
print(f"Dropping {len(cols_to_drop)} leakage columns: {cols_to_drop}")

df_cleaned = df.drop(columns=cols_to_drop)

# Kaydet
df_cleaned.to_csv(output_file, index=False)
print(f"Cleaned dataset saved to {output_file}")
print(f"Original shape: {df.shape}")
print(f"New shape: {df_cleaned.shape}")
