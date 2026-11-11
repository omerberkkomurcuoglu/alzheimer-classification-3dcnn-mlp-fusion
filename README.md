# OASIS-3 Üzerinde 3D Yapısal MRI ve Klinik Verilerle Alzheimer Hastalığının Erken Tespiti için Çok Modelli Geç Füzyon Mimarisi

[Akademik Makale Tam Metni (paper.pdf)](paper.pdf)

---

## Özet

Bu çalışma, boylamsal (longitudinal) OASIS-3 kohort veri seti kullanılarak Alzheimer hastalığının evrelerinin (Normal, Hafif Bilişsel Bozukluk ve Orta/Ağır Demans) tespiti amacıyla geliştirilen uçtan uca çok modelli bir derin öğrenme ve makine öğrenmesi sistemidir. Çalışma kapsamında 3D T1-ağırlıklı yapısal MR görüntüleri ile bilişsel ve klinik değerlendirme verileri optimize edilmiş ağırlıklı geç füzyon (weighted late-fusion) mimarisiyle bir araya getirilmiştir.

Metodolojide, literatürde sıklıkla karşılaşılan kimlik sızıntısının (data leakage) önüne geçmek amacıyla hasta bazlı `StratifiedGroupKFold` ayrıştırması uygulanmış; doğrudan teşhisi ele veren CDR-SB, MMSE gibi değişkenler elenerek hedef sızıntısı (target leakage) engellenmiştir. Modellerin hiperparametreleri Optuna ile optimize edilmiş, model kararları ise 3D Grad-CAM ve özellik önem analizleri ile açıklanabilir hale getirilmiştir.

---

## Veri Seti ve Sınıf Dağılımı

Washington Üniversitesi Knight ADRC tarafından sağlanan OASIS-3 kohortundan, yaşları 42 ile 95 arasında değişen 1.360 hastaya ait toplam 3.692 çok modelli kayıt kullanılmıştır. MR çekim tarihleri ile klinik test ziyaretleri arasında azami ±180 günlük zaman farkı toleransı uygulanmıştır.

| Sınıf | CDR Skoru | Örnek Sayısı | Oran (%) | Eğitim / Test Bölünmesi | Klinik Durum |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Normal (CN)** | 0 | 2.950 | %79.9 | 2.360 / 590 | Bilişsel olarak sağlıklı bireyler |
| **Hafif (MCI)** | 0.5 | 469 | %12.7 | 375 / 94 | Hafif bilişsel bozukluk evresi |
| **Orta / Ağır (AD)** | >= 1.0 | 273 | %7.4 | 218 / 55 | Orta ve ileri evre demans |
| **Toplam** | — | **3.692** | **%100** | **2.953 / 739** | Hasta bazlı yalıtılmış dağılım |

---

## Yöntem

### 1. Veri Hazırlama ve Ön İşleme Hattı (`scripts/preprocessing/`)

Veri hazırlama süreci adım adım takip edilebilir 10 aşamadan oluşmaktadır:
1. `01_merge_clinical_tables.ipynb`: 18 farklı UDS klinik tablosunun hasta kimliği ve ziyaret yaşı üzerinden birleştirilmesi, %20'den fazla eksik veriye sahip sütunların ayıklanması.
2. `02_impute_clinical_data.ipynb`: Eksik klinik verilerin hasta bazında ileri doldurma (forward-fill) yöntemiyle tamamlanması.
3. `03_match_mri_sessions.ipynb`: MR oturumlarının CDR etiketleri ve zaman aralıklarıyla eşleştirilmesi.
4. `04_encode_categorical_and_meds.ipynb`: Kategorik değişkenlerin kodlanması ve 2.692 farklı ilacın 16 boyutlu gömme vektörlerine dönüştürülmesi.
5. `05_merge_mri_and_clinical.ipynb`: Ön işlenmiş klinik tablo ile MR dosya yollarının birleştirilmesi.
6. `06_reduce_med_embeddings_pca.py`: İlaç gömme boyutunun PCA ile 16 boyuttan 8 bileşene indirgenmesi.
7. `07_integrate_pca_embeddings.py`: İndirgenmiş ilaç bileşenlerinin ana veri setine entegrasyonu.
8. `08_preprocess_mri_volumes_128.py`: 3D T1w MR hacimlerinin 1 mm izotropik voksel aralığına ve RAS oryantasyonuna getirilmesi, arka plan dokuların kırpılması, Z-skor normalizasyonu ve 128x128x128 boyutlarına standartlaştırılması.
9. `09_remove_target_leakage.py`: Teşhis sonucunu doğrudan işaret eden değişkenlerin (CDRSUM, CDRTOT, MMSE, tanı kodları vb.) çıkarılarak 84 temiz klinik özniteliğe ulaşılması.
10. `10_align_mri_clinical_180d.py`: MR ve klinik oturumlar arasında 180 günlük eşleşme kontrolünün uygulanması ve MR bazında tekilleştirme yapılması.

### 2. Model Mimarileri

- **3D Nörogörüntüleme Modelleri:**
  - *Custom 3D CNN:* Küçük yığın boyutlarında kararlı çalışan InstanceNorm3d ve GroupNorm katmanları, 4 aşamalı Residual bloklar, bölgesel doku kaybına odaklanan Squeeze-and-Excitation (SE) dikkat mekanizmaları ve aşamalı 3D Dropout (~8.2M parametre).
  - *ResNet50 (MedicalNet):* 23 farklı tıbbi veri setiyle önceden eğitilmiş, LayerNorm ve GELU aktivasyonlu özel sınıflandırma başlığı eklenmiş, felaketsel unutmayı (catastrophic forgetting) önlemek amacıyla 3 aşamalı kademeli katman çözme (gradual unfreezing) stratejisi ve EMA (decay=0.998) ile eğitilmiş 3D ResNet50 modeli.
- **Klinik Tablo Modelleri:**
  - 7 farklı algoritma: XGBoost, LightGBM, CatBoost, Random Forest, TabNet, FT-Transformer ve 5 katmanlı Artık Bağlantılı MLP.
  - Modellerin tamamı Optuna ve MedianPruner kullanılarak hiperparametre optimizasyonuna tabi tutulmuştur.
- **Ağırlıklı Geç Füzyon Stratejisi:**
  - Birbirinden bağımsız olarak eğitilen görüntü ve klinik modellerinin softmax olasılık çıktıları karar aşamasında birleştirilmiştir:
  $$P_{\text{final}} = \alpha \cdot P_{\text{MRI}} + (1 - \alpha) \cdot P_{\text{Klinik}}, \quad \alpha = 0.30$$
  - Yapılan ablasyon deneyleri sonucunda, klinik modele %70, 3D MR modeline %30 ağırlık verilmesinin baskın modalite problemini engellediği ve dengeli doğruluğu en üst düzeye çıkardığı belirlenmiştir.

---

## Deneysel Sonuçlar

Veri setindeki belirgin sınıf dengesizliği (%79.9 Normal) sebebiyle birincil başarım ölçütü olarak **Dengeli Doğruluk (Balanced Accuracy - BAcc)** kullanılmıştır. Tüm sonuçlar eğitimde hiç kullanılmamış, hasta bazında izole test setinde (n=739) elde edilmiştir:

| Model Konfigürasyonu | Doğruluk (%) | Dengeli Doğruluk (%) | Kesinlik (%) | Duyarlılık (%) | F1-Skoru (%) | ROC-AUC | 5-Katlı ÇD BAcc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet50 + XGBoost (En İyi BAcc)** | 87.42 | **78.13** | 74.79 | 78.13 | **76.34** | **0.964** | **76.8 ± 2.1** |
| **Custom3DCNN + Random Forest (En İyi Acc)** | **88.76** | 75.64 | **77.60** | 75.64 | 75.84 | 0.951 | 74.2 ± 2.8 |
| **ResNet50 + LightGBM** | 86.29 | 77.15 | 72.20 | 77.15 | 74.42 | 0.958 | 75.8 ± 2.3 |
| **XGBoost (Yalnız Klinik)** | 86.74 | 76.84 | 72.45 | 76.84 | 74.45 | 0.961 | 75.4 ± 3.1 |
| **ResNet50 + FT-Transformer** | 84.94 | 76.70 | — | — | — | — | 75.1 ± 3.4 |
| **Custom3DCNN + XGBoost** | 85.84 | 75.96 | — | — | — | — | 74.6 ± 2.9 |
| **ResNet50 + CatBoost** | 86.97 | 74.35 | 74.21 | 74.35 | 74.16 | 0.961 | 73.2 ± 3.1 |
| **Custom3DCNN (Yalnız MRI)** | 68.09 | 55.01 | — | — | — | 0.813 | 54.2 ± 5.1 |
| **ResNet50 (Yalnız MRI)** | 71.24 | 50.33 | — | — | — | 0.854 | 49.6 ± 5.8 |

---

## Açıklanabilirlik ve Görsel Analizler

### Grad-CAM Odak Haritaları (3D ResNet50)

Modelin MR görüntülerindeki karar odak noktaları incelendiğinde, Alzheimer hastalığına işaret eden klinik biyobelirteçlerle tam uyum sağladığı görülmektedir:
- **Normal (CN):** Dikkatin beyin geneline dengeli yayıldığı, lokal bir atrofi bölgesine yoğunlaşmadığı gözlenmektedir.
- **Hafif Bilişsel Bozukluk (MCI):** Aktivasyonun erken dönem hipokampal alanlar ve temporal lob üzerinde yoğunlaştığı tespit edilmiştir.
- **Orta / Ağır Demans (AD):** Model odağının genişlemiş lateral ventriküllere ve belirgin kortikal atrofi bölgelerine kaydığı görülmektedir.

| Normal (CN) | Hafif Bilişsel Bozukluk (MCI) | Orta / Ağır Demans (AD) |
| :---: | :---: | :---: |
| ![Normal](graphics/gradcam_Normal_2.png) | ![MCI](graphics/gradcam_Mild_2.png) | ![AD](graphics/gradcam_Moderate_Severe_2.png) |

### Performans Eğrileri

| Karmaşıklık Matrisi (ResNet50 + XGBoost) | ROC Eğrileri (ResNet50 + XGBoost) |
| :---: | :---: |
| ![Karmaşıklık Matrisi](graphics/fusion_ResNet50_XGBoost_cm.png) | ![ROC Eğrisi](graphics/fusion_ResNet50_XGBoost_roc.png) |

---

## Proje Dizin Yapısı

```text
Bitirme_Projesi/
├── paper.pdf                                           # Çalışmanın tam metin akademik makalesi
├── graphics/                                           # Metrik grafikleri (ROC, CM, Grad-CAM, Özellik Önemi)
├── scripts/
│   ├── preprocessing/                                  # 10 adımlı veri ön işleme ve sızıntı önleme hattı
│   │   ├── 01_merge_clinical_tables.ipynb
│   │   ├── 02_impute_clinical_data.ipynb
│   │   ├── 03_match_mri_sessions.ipynb
│   │   ├── 04_encode_categorical_and_meds.ipynb
│   │   ├── 05_merge_mri_and_clinical.ipynb
│   │   ├── 06_reduce_med_embeddings_pca.py
│   │   ├── 07_integrate_pca_embeddings.py
│   │   ├── 08_preprocess_mri_volumes_128.py
│   │   ├── 09_remove_target_leakage.py
│   │   └── 10_align_mri_clinical_180d.py
│   │
│   └── train/
│       ├── clinical/                                   # Klinik modellerin eğitim, Optuna ve değerlendirme kodları
│       │   ├── configs/                                # Optuna ile bulunan en iyi hiperparametreler (*.json)
│       │   ├── xgboost_{optuna_tuning,train,evaluate}.py
│       │   ├── lightgbm_{optuna_tuning,train,evaluate}.py
│       │   ├── catboost_{optuna_tuning,train,evaluate}.py
│       │   ├── random_forest_{optuna_tuning,train,evaluate}.py
│       │   ├── tabnet_{optuna_tuning,train,evaluate}.py
│       │   ├── ft_transformer_{optuna_tuning,train,evaluate}.py
│       │   └── mlp_{optuna_tuning,train,evaluate}.py
│       │
│       ├── mr/                                         # 3D MRI model eğitimleri ve Grad-CAM analizleri
│       │   ├── custom_3dcnn_{train,evaluate}.py
│       │   ├── resnet50_medicalnet_{optuna_tuning,train,evaluate}.py
│       │   └── gradcam_3d_visualize.py
│       │
│       ├── fusion/                                     # Geç füzyon değerlendirmesi ve modalite ablasyon analizleri
│       │   ├── evaluate_late_fusion.py
│       │   ├── ablation_modality_analysis.py
│       │   └── fusion_results.csv
│       │
│       ├── custom_model.py                             # Özgün 3D CNN model mimarisi
│       └── mri_pipeline.py                             # 3D MRI veri yükleme ve işleme araçları
│
├── mobil-app/                                          # Karar destek sistemi mobil & masaüstü uygulaması
│   ├── backend/                                        # FastAPI multimodal tahmin sunucusu
│   ├── frontend/                                       # Flutter kullanıcı arayüzü
│   └── inference/                                      # Test amaçlı örnek hasta verileri
├── website/                                            # Streamlit tabanlı web arayüzü
│   └── scripts/
│       ├── app.py
│       └── utils.py
├── .gitignore
└── README.md
```

---

## Veri Seti Referansı

Bu çalışmada kullanılan veriler Washington Üniversitesi Knight ADRC tarafından sağlanan OASIS-3 projesine aittir:

```bibtex
@article{lamontagne2019oasis3,
  title={OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset for Normal Aging and Alzheimer Disease},
  author={LaMontagne, Pamela J and Benzinger, Tammie LS and Morris, John C and Keefe, Sarah and Hornbeck, Russ and Xiong, Chengjie and others},
  journal={medRxiv},
  year={2019},
  publisher={Cold Spring Harbor Laboratory Preprints},
  doi={10.1101/2019.12.13.19014902}
}
```
