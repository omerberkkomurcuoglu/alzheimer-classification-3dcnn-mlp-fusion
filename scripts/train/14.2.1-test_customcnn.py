"""
Custom 3D CNN Model Test Script
Eğitilmiş Custom3DCNN ağırlıklarını test eder.
Validation + Test: Confusion Matrix, Classification Report, AUC-ROC Eğrileri
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from mri_pipeline import coerce_memory_label, filter_existing_mri_rows, load_normalized_mri_volume

# ============================================================
# 1. DATASET
# ============================================================
class MRIOnlyDataset(Dataset):
    """Sadece MRI görüntülerini kullanan dataset"""
    
    def __init__(self, root_dir, data_csv):
        self.root_dir = root_dir
        self.data_csv = filter_existing_mri_rows(root_dir, data_csv)
        self.label_map = {0: 0, 0.5: 1, 1: 2, 2: 2}

    def __len__(self):
        return len(self.data_csv)
    
    def __getitem__(self, idx):
        row = self.data_csv.iloc[idx]
        session_id = row["path"]
        file_path = os.path.join(self.root_dir, f"{session_id}.npy")

        data = load_normalized_mri_volume(file_path)
        label = coerce_memory_label(row['memory'])

        return torch.from_numpy(data).unsqueeze(0), torch.tensor(label, dtype=torch.long)

# ============================================================
# 2. MODEL
# ============================================================
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from custom_model import Custom3DCNN

# ============================================================
# 3. DEĞERLENDİRME FONKSİYONU
# ============================================================
def evaluate_set(model, loader, device, set_name, class_names, save_prefix):
    """
    Bir set (val veya test) için:
    - Classification Report
    - Confusion Matrix
    - AUC-ROC Eğrileri
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"{set_name}"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    num_classes = len(class_names)
    
    # --- Metrikleri Yazdır ---
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    
    print(f"\n{'='*60}")
    print(f"{set_name} RESULTS (3 CLASSES)")
    print(f"{'='*60}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Balanced Accuracy: {bal_acc*100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # --- Confusion Matrix ---
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
    
    # --- AUC-ROC Eğrileri ---
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
    
    # Micro-average
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
    
    # Veri yolları
    processed_dir = "C:\\Users\\berkk\\Desktop\\Bitirme_Projesi\\processed_dataset_128"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_csv = pd.read_csv(os.path.join(script_dir, "12-data_leakage.csv"))
    
    print(f"Toplam veri sayısı: {len(data_csv)}")
    
    # Veri bölme (16-train_simplecnn.py ile aynı split)
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_inds, test_inds = next(splitter_test.split(data_csv, groups=data_csv['OASISID']))

    train_val_df = data_csv.iloc[train_val_inds].reset_index(drop=True)
    test_df = data_csv.iloc[test_inds].reset_index(drop=True)

    # Validation split (aynı train scriptiyle)
    val_ratio = 15 / 85
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_inds, val_inds = next(splitter_val.split(train_val_df, groups=train_val_df['OASISID']))
    
    val_df = train_val_df.iloc[val_inds].reset_index(drop=True)
    
    # Datasetler
    val_dataset = MRIOnlyDataset(processed_dir, val_df)
    test_dataset = MRIOnlyDataset(processed_dir, test_df)
    
    print(f"Validation dataset: {len(val_dataset)}")
    print(f"Test dataset: {len(test_dataset)}")
    
    # DataLoaders
    BATCH_SIZE = 10
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    model = Custom3DCNN(num_classes=3, dropout=0.2)
    model.to(device)
    
    # Ağırlık Yükleme
    weights_path = "C:\\Users\\berkk\\Desktop\\Bitirme_Projesi\\scripts\\weights\\Custom3DCNN_best_balanced.pth"
    
    if not os.path.exists(weights_path):
        print(f"\n❌ {weights_path} bulunamadı!")
        print("Lütfen önce '16-train_simplecnn.py' ile modeli eğitin.")
        exit()
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"\n✅ Model başarıyla yüklendi: {weights_path}")
    except RuntimeError as e:
        print(f"\n❌ Model yükleme hatası: {e}")
        exit()
    
    model.eval()
    class_names = ['Normal', 'Mild', 'Moderate/Severe']
    
    # --- VALIDATION ---
    print("\n🚀 Validation Evaluation...")
    val_acc, val_bal, val_aucs = evaluate_set(
        model, val_loader, device,
        set_name="VALIDATION",
        class_names=class_names,
        save_prefix="customcnn_val"
    )
    
    # --- TEST ---
    print("\n🚀 Test Evaluation...")
    test_acc, test_bal, test_aucs = evaluate_set(
        model, test_loader, device,
        set_name="TEST",
        class_names=class_names,
        save_prefix="customcnn_test"
    )
    
    # --- ÖZET ---
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
