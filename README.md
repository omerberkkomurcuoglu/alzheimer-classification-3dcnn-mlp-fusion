# 🧠 3D Beyin MRI ve Klinik Verilerden Alzheimer Hastalığı Sınıflandırması

OASIS-3 longitudinal nörögörüntüleme veri seti kullanılarak, 3D yapısal MRI taramaları ve klinik (tabular) verilerden Alzheimer hastalığı şiddetini sınıflandıran derin öğrenme projesi.

## 📋 Genel Bakış

Bu proje, **3 sınıflı** Alzheimer tespit görevi için üç farklı sınıflandırma modeli içermektedir:

| Sınıf | CDR Skoru | Açıklama |
|-------|-----------|----------|
| **Normal** | 0 | Bilişsel olarak normal |
| **Hafif** | 0.5 | Çok hafif / hafif bilişsel bozukluk |
| **Orta/Ağır** | 1–2 | Orta-ağır demans |

### Modeller

| Model | Girdi | Mimari | Öne Çıkan Özellikler |
|-------|-------|--------|---------------------|
| **MLP** | Yalnızca klinik veri | Artık bağlantılı 3 bloklu MLP | Aşamalı CE → Focal Loss, permutation feature importance |
| **ResNet50** | Yalnızca 3D MRI | MedicalNet ResNet50 (23 medikal veri setiyle ön eğitimli) | 3 fazlı eğitim (ısınma → ince ayar → SWA), TTA |
| **Fusion** | 3D MRI + Klinik veri | Custom3DCNN + MLP, çapraz-modal dikkat | Modalite dropout, derin denetim, dikkat kapıları |

## 🗂️ Proje Yapısı

```
Bitirme_Projesi/
├── scripts/
│   ├── preprocessing/              # Veri hazırlama hattı
│   │   ├── 1-merge_csv.ipynb               # Ham CSV dosyalarını birleştirme
│   │   ├── 2-filling_data.ipynb            # Eksik verileri doldurma
│   │   ├── 3-mr_process.ipynb              # MRI görselleştirme ve inceleme
│   │   ├── 4-files_set.ipynb               # MR oturumlarını CDR etiketleriyle eşleme
│   │   ├── 7-categorical.ipynb             # Kategorik özellikleri kodlama
│   │   ├── 9-merge_mr_csv.ipynb            # MRI yolları ile klinik verileri birleştirme
│   │   ├── 10.1-preprocess_dataset_128.py  # MRI ön işleme ve 128³ boyutlandırma
│   │   └── 12-remove_leakage.py            # Veri sızıntısı önleme (hasta bazlı)
│   │
│   ├── train/                      # Model eğitimi ve değerlendirme
│   │   ├── 13-train_mlp_only.py            # MLP (yalnızca klinik veri)
│   │   ├── 14.1-train_simplecnn.py         # Özel 3D CNN
│   │   ├── 14.3-train_resnet50.py          # MedicalNet ResNet50
│   │   ├── 15.0-CustomCnn_MLP_Fusion.py    # CNN + MLP Füzyon modeli
│   │   ├── 14.2-gradcam_visualize.py       # GradCAM görselleştirme
│   │   ├── 14.3.1-test_medicalnet_resnet50.py  # ResNet50 test değerlendirmesi
│   │   ├── 15.1-test_fusion.py             # Füzyon model test değerlendirmesi
│   │   ├── 15.2-analyze_modality_importance.py # Modalite ablasyon çalışması
│   │   ├── custom_model.py                 # Custom3DCNN mimarisi
│   │   └── mri_pipeline.py                 # MRI yükleme ve normalizasyon araçları
│   │
│   ├── website/                    # Streamlit web uygulaması
│   │   └── scripts/
│   │       ├── app.py                      # Ana Streamlit uygulaması
│   │       └── utils.py                    # Yardımcı fonksiyonlar
│   │
│   └── repo/                       # Yayın için hazır scriptler
│
├── Csv_Files/                      # Ham OASIS-3 UDS klinik verileri (repo'da yok)
├── processed_dataset_128/          # Ön işlenmiş 128³ MRI hacimleri (repo'da yok)
└── .gitignore
```

## 🔬 Teknik Detaylar

### Veri Hattı
1. **Ham Veri**: OASIS-3 UDS klinik değerlendirmeleri + T1w yapısal MRI taramaları
2. **MRI Ön İşleme**: Kafatası çıkarma → 128×128×128 yeniden örnekleme → z-skor normalizasyonu
3. **Klinik Özellikler**: Bilişsel testler (UDS bataryası) ve demografik bilgilerden ~72 özellik
4. **Veri Bölme**: %70/%15/%15 (eğitim/doğrulama/test), veri sızıntısını önlemek için **hasta bazlı gruplama**

### ResNet50 Eğitim Stratejisi
- **Faz 1**: Baş katman ısınması (omurga dondurulmuş, 15 epoch)
- **Faz 2a**: Yalnızca layer4 ince ayar (20 epoch, diferansiyel öğrenme oranı)
- **Faz 2b**: layer3 + layer4 ince ayar (30 epoch)
- **Faz 3**: Stokastik Ağırlık Ortalaması / SWA (5 epoch)
- **Değerlendirme**: Test Zamanı Artırma / TTA (5 katlı)

### Füzyon Modeli
- **Çapraz-Modal Dikkat Kapıları**: MRI ve klinik özelliklerin önemini öğrenir
- **Asimetrik Modalite Dropout**: %70 klinik / %30 MRI düşürme oranı
- **Derin Denetim**: CNN ve MLP dallarında yardımcı kayıp fonksiyonları

## ⚙️ Gereksinimler

```
torch
monai
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
huggingface_hub
streamlit
```

## 🚀 Kullanım

### 1. Ön İşleme
```bash
python scripts/preprocessing/10.1-preprocess_dataset_128.py
python scripts/preprocessing/12-remove_leakage.py
```

### 2. Eğitim
```bash
# MLP (yalnızca klinik veri)
python scripts/train/13-train_mlp_only.py

# ResNet50 (yalnızca 3D MRI)
python scripts/train/14.3-train_resnet50.py

# Füzyon (MRI + Klinik)
python scripts/train/15.0-CustomCnn_MLP_Fusion.py
```

### 3. Değerlendirme
```bash
python scripts/train/14.3.1-test_medicalnet_resnet50.py
python scripts/train/15.1-test_fusion.py
```

### 4. Web Uygulaması
```bash
streamlit run scripts/website/scripts/app.py
```

## 📊 Veri Seti

Bu proje [OASIS-3](https://www.oasis-brains.org/) veri setini kullanmaktadır:

> **OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset for Normal Aging and Alzheimer Disease**
> LaMontagne et al., 2019

> [!NOTE]
> Ön işlenmiş MRI verileri (`processed_dataset_128/`) ve model ağırlıkları (`scripts/weights/`) boyutları nedeniyle bu repoda yer almamaktadır.

## 📄 Lisans

Bu proje bir bitirme tezi kapsamında geliştirilmiştir.
