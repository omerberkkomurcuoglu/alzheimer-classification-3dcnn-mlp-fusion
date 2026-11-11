# Multimodal Late-Fusion Architecture for Early Alzheimer's Detection on OASIS-3

A Comprehensive Study on 3D Structural MRI and Clinical-Cognitive Data

[Read Full Academic Paper (paper.pdf)](paper.pdf)

---

## Abstract

This repository presents an end-to-end multimodal machine learning framework for classifying Alzheimer's Disease stages (Cognitively Normal, Mild Cognitive Impairment, and Moderate/Severe Dementia) using the longitudinal OASIS-3 cohort. The framework integrates 3D T1-weighted structural MRI scans and tabular clinical/neuropsychological assessments via an optimized weighted late-fusion architecture.

Key design considerations include strict prevention of both identity leakage (via patient-level `StratifiedGroupKFold`) and target leakage (by systematically removing direct diagnostic proxies such as CDR-SB and MMSE), hyperparameter optimization via Optuna with pruning, and model interpretability via 3D Grad-CAM and feature importance analysis.

---

## Dataset and Cohort Characteristics

Data is sourced from the Open Access Series of Imaging Studies (OASIS-3) released by Washington University Knight ADRC. The final cohort comprises 3,692 multimodal records across 1,360 unique participants (ages 42–95) matched within a +/- 180-day tolerance window between MRI scanning and clinical visit dates.

| Class | Clinical Dementia Rating (CDR) | Total Samples | Proportion (%) | Train / Test Split | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Normal (CN)** | 0 | 2,950 | 79.9% | 2,360 / 590 | Cognitively unimpaired controls |
| **Mild (MCI)** | 0.5 | 469 | 12.7% | 375 / 94 | Mild cognitive impairment |
| **Moderate / Severe (AD)** | >= 1.0 | 273 | 7.4% | 218 / 55 | Moderate to severe Alzheimer's dementia |
| **Total** | — | **3,692** | **100%** | **2,953 / 739** | Patient-isolated split (zero leakage) |

---

## Methodology

### 1. Data Preprocessing Pipeline (`scripts/preprocessing/`)

The data curation follows a reproducible 10-step sequence:
1. `01_merge_clinical_tables.ipynb`: Combines 18 OASIS-3 UDS tables using `OASISID` and `age_at_visit`, filtering features with >20% missing values.
2. `02_impute_clinical_data.ipynb`: Imputes missing clinical fields via patient-specific forward-filling.
3. `03_match_mri_sessions.ipynb`: Maps MRI scan sessions to corresponding CDR assessments and temporal intervals.
4. `04_encode_categorical_and_meds.ipynb`: Categorical feature encoding; builds 16-dimensional embedding vectors for 2,692 unique medication names.
5. `05_merge_mri_and_clinical.ipynb`: Merges preprocessed clinical features with corresponding MRI file paths.
6. `06_reduce_med_embeddings_pca.py`: Reduces medication embedding dimensionality from 16D to 8 components using PCA.
7. `07_integrate_pca_embeddings.py`: Integrates PCA-reduced medication features back into the primary tabular dataset.
8. `08_preprocess_mri_volumes_128.py`: Resamples 3D T1w MRI to 1 mm isotropic spacing, aligns to RAS orientation, crops background tissue, clips intensities (1st–99.5th percentile), normalizes via Z-score, and standardizes to 128x128x128 volumes.
9. `09_remove_target_leakage.py`: Eliminates direct diagnostic proxies (CDRSUM, CDRTOT, MMSE, diagnostic code flags) to establish 84 leakage-free predictor variables.
10. `10_align_mri_clinical_180d.py`: Enforces the <= 180 days temporal matching threshold and deduplicates records per MRI session.

### 2. Evaluated Architectures

- **3D Neuroimaging Branch:**
  - *Custom 3D CNN:* Residual architecture with InstanceNorm3d, GroupNorm, Squeeze-and-Excitation (SE) channel attention blocks, and progressive spatial dropout (~8.2M parameters).
  - *ResNet50 (MedicalNet):* 3D ResNet50 pre-trained on 23 medical datasets, fine-tuned using a 3-phase gradual unfreezing schedule with LayerNorm, GELU, mixed precision, and Exponential Moving Average (EMA, decay=0.998).
- **Clinical/Tabular Branch:**
  - 7 algorithms evaluated: XGBoost, LightGBM, CatBoost, Random Forest, TabNet, FT-Transformer, and Residual MLP.
  - All algorithms tuned via Optuna using `MedianPruner`.
- **Weighted Late-Fusion Strategy:**
  - Softmax probabilities from independent imaging and tabular models are combined at inference time:
  $$P_{\text{final}} = \alpha \cdot P_{\text{MRI}} + (1 - \alpha) \cdot P_{\text{Clinical}}, \quad \alpha = 0.30$$
  - Ablation experiments demonstrate that giving 70% weight to tabular clinical data and 30% to structural MRI mitigates modality dominance while maximizing balanced accuracy.

---

## Experimental Results

Due to severe class imbalance (79.9% Normal), **Balanced Accuracy (BAcc)** serves as the primary evaluation metric. All models were evaluated on an unseen, patient-isolated test set (n=739):

| Model Configuration | Accuracy (%) | Balanced Acc (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC | 5-Fold CV BAcc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet50 + XGBoost (Best BAcc)** | 87.42 | **78.13** | 74.79 | 78.13 | **76.34** | **0.964** | **76.8 +/- 2.1** |
| **Custom3DCNN + Random Forest (Best Acc)** | **88.76** | 75.64 | **77.60** | 75.64 | 75.84 | 0.951 | 74.2 +/- 2.8 |
| **ResNet50 + LightGBM** | 86.29 | 77.15 | 72.20 | 77.15 | 74.42 | 0.958 | 75.8 +/- 2.3 |
| **XGBoost (Clinical Only)** | 86.74 | 76.84 | 72.45 | 76.84 | 74.45 | 0.961 | 75.4 +/- 3.1 |
| **ResNet50 + FT-Transformer** | 84.94 | 76.70 | — | — | — | — | 75.1 +/- 3.4 |
| **Custom3DCNN + XGBoost** | 85.84 | 75.96 | — | — | — | — | 74.6 +/- 2.9 |
| **ResNet50 + CatBoost** | 86.97 | 74.35 | 74.21 | 74.35 | 74.16 | 0.961 | 73.2 +/- 3.1 |
| **Custom3DCNN (MRI Only)** | 68.09 | 55.01 | — | — | — | 0.813 | 54.2 +/- 5.1 |
| **ResNet50 (MRI Only)** | 71.24 | 50.33 | — | — | — | 0.854 | 49.6 +/- 5.8 |

---

## Interpretability and Saliency Analysis

### Grad-CAM Visualizations (3D ResNet50)

The model localizes clinically established radiological biomarkers of Alzheimer's Disease progression:
- **Normal (CN):** Activations remain diffuse across cortical parenchyma without focal concentration.
- **Mild Cognitive Impairment (MCI):** Early focal activation around the medial temporal lobes and hippocampal structures.
- **Moderate / Severe (AD):** Pronounced saliency concentrated in enlarged lateral ventricles and severe cortical atrophy zones.

| Normal (CN) | Mild Cognitive Impairment (MCI) | Moderate / Severe Dementia |
| :---: | :---: | :---: |
| ![Normal](graphics/gradcam_Normal_2.png) | ![Mild](graphics/gradcam_Mild_2.png) | ![Severe](graphics/gradcam_Moderate_Severe_2.png) |

### Performance Curves

| Confusion Matrix (ResNet50 + XGBoost) | ROC Curves (ResNet50 + XGBoost) |
| :---: | :---: |
| ![Confusion Matrix](graphics/fusion_ResNet50_XGBoost_cm.png) | ![ROC Curve](graphics/fusion_ResNet50_XGBoost_roc.png) |

---

## Repository Structure

```text
Bitirme_Projesi/
├── paper.pdf                                           # Full academic manuscript
├── graphics/                                           # Evaluation figures (ROC, CM, Grad-CAM, Feature Importance)
├── scripts/
│   ├── preprocessing/                                  # 10-step data preparation and leakage mitigation
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
│       ├── clinical/                                   # Tabular model training, Optuna tuning, and evaluation
│       │   ├── configs/                                # Best hyperparameters discovered by Optuna (*.json)
│       │   ├── xgboost_{optuna_tuning,train,evaluate}.py
│       │   ├── lightgbm_{optuna_tuning,train,evaluate}.py
│       │   ├── catboost_{optuna_tuning,train,evaluate}.py
│       │   ├── random_forest_{optuna_tuning,train,evaluate}.py
│       │   ├── tabnet_{optuna_tuning,train,evaluate}.py
│       │   ├── ft_transformer_{optuna_tuning,train,evaluate}.py
│       │   └── mlp_{optuna_tuning,train,evaluate}.py
│       │
│       ├── mr/                                         # 3D MRI models and Grad-CAM visualization
│       │   ├── custom_3dcnn_{train,evaluate}.py
│       │   ├── resnet50_medicalnet_{optuna_tuning,train,evaluate}.py
│       │   └── gradcam_3d_visualize.py
│       │
│       ├── fusion/                                     # Late fusion evaluation and ablation analysis
│       │   ├── evaluate_late_fusion.py
│       │   ├── ablation_modality_analysis.py
│       │   └── fusion_results.csv
│       │
│       ├── custom_model.py                             # Custom 3D CNN PyTorch definition
│       └── mri_pipeline.py                             # MRI volume loading and processing utilities
│
├── mobil-app/                                          # Clinical decision support application
│   ├── backend/                                        # FastAPI multimodal inference server
│   ├── frontend/                                       # Flutter mobile / desktop interface
│   └── inference/                                      # Sample patient data for testing
├── website/                                            # Streamlit-based web deployment
│   └── scripts/
│       ├── app.py
│       └── utils.py
├── .gitignore
└── README.md
```

---

## Citation

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
