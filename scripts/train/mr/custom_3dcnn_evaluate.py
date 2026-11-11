import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import importlib.util

# --- Add parent dir to path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

# --- Import from the training script to ensure exactly matching splits and definitions ---
try:
    # Use dynamic import to handle file names that might have changed
    spec = importlib.util.spec_from_file_location("trainer", os.path.join(SCRIPT_DIR, "1-train_CustomCNN_NoDoubleWeight.py"))
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
except Exception as e:
    print(f"Error importing objects from training script: {e}")
    sys.exit(1)

# Import the model
from custom_model import Custom3DCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_set(model, loader, device, set_name, class_names, save_prefix, use_tta=False):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"{set_name}"):
            images = images.to(device)
            
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                if use_tta:
                    probs = trainer.predict_with_tta(model, images)
                else:
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    
    tag = " (TTA)" if use_tta else ""
    print(f"\n{'='*60}")
    print(f"{set_name}{tag} RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Balanced Accuracy: {bal_acc*100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion Matrix (Improved & Professional)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 13, "weight": "bold"},
                cbar_kws={'label': 'Number of Samples'})
    
    plt.title(f'{set_name}\nAccuracy: {acc*100:.2f}% | Balanced Acc: {bal_acc*100:.2f}%', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=25, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, f"{save_prefix}_confusion_matrix.png"), dpi=200)
    plt.close()

    from sklearn.metrics import f1_score
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # ROC Curve (Improved & Professional)
    y_true_bin = label_binarize(all_labels, classes=range(len(class_names)))
    n_classes = y_true_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    
    # Plot classes with specific colors from reference
    colors = ['#2196F3', '#FF9800', '#E91E63'] # Blue, Orange, Pink/Magenta
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

    # Plot micro-average as black dotted line
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-avg (AUC = {roc_auc["micro"]:.3f})',
             color='black', linestyle=':', linewidth=3)

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curve: {set_name}\nF1: {macro_f1*100:.1f}% | BAcc: {bal_acc*100:.1f}%', 
              fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)
    plt.grid(True, linestyle='-', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, f"{save_prefix}_roc_curve.png"), dpi=200)
    plt.close()
    
    return acc, bal_acc
    
    return acc, bal_acc

def main():
    print(f"Using device: {DEVICE}")
    
    # Get CFG from trainer
    CFG = trainer.CFG
    
    # Try customcnn.pth first, otherwise fallback to the best weights from 1.2
    custom_path = os.path.join(CFG["weights_dir"], "customcnn.pth")
    if os.path.exists(custom_path):
        WEIGHTS_PATH = custom_path
        print(f"\nTesting explicitly against weights: customcnn.pth")
    else:
        WEIGHTS_NAME = "Custom3DCNN_v2_ce_unweighted_1_2_best_balanced.pth"
        WEIGHTS_PATH = os.path.join(CFG["weights_dir"], WEIGHTS_NAME)
        print(f"\nTesting explicitly against weights: {WEIGHTS_NAME}")
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERROR: Cannot find weights path: {WEIGHTS_PATH}")
        sys.exit(1)
        
    # 1. Dataset loading exactly matching the trainer
    raw_df = pd.read_csv(CFG["csv_path"])
    df = trainer.filter_existing_mri_rows(CFG["processed_dir"], raw_df)
    
    # We only need the test dataset dataframe. make_group_splits returns (train_val, val, test)
    _, _, test_df = trainer.make_group_splits(df)
    print(f"Test samples (exactly matched to original split): {len(test_df)}")
    
    test_ds = trainer.BrainMRIDataset(CFG["processed_dir"], test_df, augment=False)
    test_loader = DataLoader(test_ds, batch_size=CFG["test_batch_size"], shuffle=False, 
                             num_workers=0, pin_memory=True)
                             
    # 2. Model initialization & Weight Loading
    # Use standard load logic matching classifier implementation
    model = trainer.CustomCNNClassifier(CFG["num_classes"], CFG["dropout"])
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.to(DEVICE)
    print("Model weights loaded successfully.")
    
    # 3. Evaluation
    evaluate_set(model, test_loader, DEVICE, "CustomCNN Test", CFG["class_names"], "1.1_test_customcnn", use_tta=False)
    
    print(f"\n[OK] All tests complete. Confusion matrices and ROC curves are saved in {SCRIPT_DIR}\n")

if __name__ == "__main__":
    main()
