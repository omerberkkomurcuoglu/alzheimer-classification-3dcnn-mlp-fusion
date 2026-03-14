# 🧠 Alzheimer's Disease Classification from 3D Brain MRI & Clinical Data

A deep learning pipeline for classifying Alzheimer's disease severity from 3D structural MRI scans and clinical (tabular) data using the **OASIS-3** longitudinal neuroimaging dataset.

## 📋 Overview

This project implements three complementary classification models for a **3-class** Alzheimer's detection task:

| Class | CDR Score | Description |
|-------|-----------|-------------|
| **Normal** | 0 | Cognitively normal |
| **Mild** | 0.5 | Very mild / mild cognitive impairment |
| **Moderate/Severe** | 1–2 | Moderate to severe dementia |

### Models

| Model | Input | Architecture | Key Features |
|-------|-------|-------------|--------------|
| **MLP** | Clinical data only | 3-block MLP with residual connections | Staged CE → Focal Loss, permutation feature importance |
| **ResNet50** | 3D MRI only | MedicalNet ResNet50 (pretrained on 23 medical datasets) | 3-phase training (warmup → fine-tune → SWA), TTA |
| **Fusion** | 3D MRI + Clinical data | Custom3DCNN + MLP with cross-modal attention | Modality dropout, deep supervision, attention gates |

## 🗂️ Project Structure

```
Bitirme_Projesi/
├── scripts/
│   ├── preprocessing/          # Data preparation pipeline
│   │   ├── 1-merge_csv.ipynb           # Merge raw OASIS-3 CSV files
│   │   ├── 2-filling_data.ipynb        # Handle missing values
│   │   ├── 3-mr_process.ipynb          # MRI visualization & exploration
│   │   ├── 4-files_set.ipynb           # Map MR sessions to CDR labels
│   │   ├── 7-categorical.ipynb         # Encode categorical features
│   │   ├── 9-merge_mr_csv.ipynb        # Merge MRI paths with clinical data
│   │   ├── 10.1-preprocess_dataset_128.py  # Preprocess & resize MRI to 128³
│   │   └── 12-remove_leakage.py        # Remove data leakage (patient-level)
│   │
│   ├── train/                  # Model training & evaluation
│   │   ├── 13-train_mlp_only.py        # MLP (clinical data only)
│   │   ├── 14.1-train_simplecnn.py     # Custom 3D CNN
│   │   ├── 14.3-train_resnet50.py      # MedicalNet ResNet50
│   │   ├── 15.0-CustomCnn_MLP_Fusion.py # CNN + MLP Fusion model
│   │   ├── 14.2-gradcam_visualize.py   # GradCAM visualization
│   │   ├── 14.3.1-test_medicalnet_resnet50.py  # ResNet50 test evaluation
│   │   ├── 15.1-test_fusion.py         # Fusion model test evaluation
│   │   ├── 15.2-analyze_modality_importance.py # Modality ablation study
│   │   ├── custom_model.py             # Custom3DCNN architecture
│   │   └── mri_pipeline.py             # MRI loading & normalization utils
│   │
│   ├── website/                # Streamlit web application
│   │   └── scripts/
│   │       ├── app.py                  # Main Streamlit app
│   │       └── utils.py                # Utility functions
│   │
│   └── repo/                   # Publication-ready scripts
│
├── Csv_Files/                  # Raw OASIS-3 UDS clinical data (not tracked)
├── processed_dataset_128/      # Preprocessed 128³ MRI volumes (not tracked)
└── .gitignore
```

## 🔬 Technical Details

### Data Pipeline
1. **Raw Data**: OASIS-3 UDS clinical assessments + T1w structural MRI scans
2. **MRI Preprocessing**: Skull-stripped → resampled to 128×128×128 → z-score normalized
3. **Clinical Features**: ~72 features from cognitive tests (UDS battery), demographics
4. **Data Split**: 70/15/15 (train/val/test) with **patient-level grouping** to prevent data leakage

### ResNet50 Training Strategy
- **Phase 1**: Head warmup (backbone frozen, 15 epochs)
- **Phase 2a**: Fine-tune layer4 only (20 epochs, differential LR)
- **Phase 2b**: Fine-tune layer3 + layer4 (30 epochs)
- **Phase 3**: Stochastic Weight Averaging (5 epochs)
- **Evaluation**: Test-Time Augmentation (5-fold)

### Fusion Model
- **Cross-Modal Attention Gates**: Learns to weight MRI vs clinical features
- **Asymmetric Modality Dropout**: 70% clinical / 30% MRI drop rate
- **Deep Supervision**: Auxiliary losses on CNN and MLP branches

## ⚙️ Requirements

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

## 🚀 Usage

### 1. Preprocessing
```bash
python scripts/preprocessing/10.1-preprocess_dataset_128.py
python scripts/preprocessing/12-remove_leakage.py
```

### 2. Training
```bash
# MLP (clinical data only)
python scripts/train/13-train_mlp_only.py

# ResNet50 (3D MRI only)
python scripts/train/14.3-train_resnet50.py

# Fusion (MRI + Clinical)
python scripts/train/15.0-CustomCnn_MLP_Fusion.py
```

### 3. Evaluation
```bash
python scripts/train/14.3.1-test_medicalnet_resnet50.py
python scripts/train/15.1-test_fusion.py
```

### 4. Web Application
```bash
streamlit run scripts/website/scripts/app.py
```

## 📊 Dataset

This project uses the [OASIS-3](https://www.oasis-brains.org/) dataset:

> **OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset for Normal Aging and Alzheimer Disease**  
> LaMontagne et al., 2019

> [!NOTE]
> The preprocessed MRI data (`processed_dataset_128/`) and model weights (`scripts/weights/`) are not included in this repository due to their large size.

## 📄 License

This project was developed as a graduation thesis (Bitirme Projesi).
