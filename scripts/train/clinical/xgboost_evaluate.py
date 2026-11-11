
import os
import random
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import label_binarize
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# 1. AYARLAR & SEED
# ============================================================
CFG = {
    "seed": 42,
    "num_classes": 3,
    "class_names": ["Normal", "Mild", "Moderate/Severe"],
    "csv_path": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\0-data_leakage_optimized.csv",
    "weights_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\weights",
    "graphs_dir": r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\clinical\graphs\xgboost",
    "model_name": "XGBoost_Clinical_Optuna",
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seed sabitlendi: {seed}")

set_seed(CFG["seed"])

# ============================================================
# 2. YARDIMCI FONKSİYONLAR (Train scripti ile uyumlu)
# ============================================================
def make_group_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # --- ALIGN WITH MRI DATASET ---
    import os
    mri_csv_path = r"C:\Users\berkk\Desktop\Bitirme_Projesi\scripts\train\mr\0-data_leakage_mri_clean_180.csv"
    processed_dir = r"C:\Users\berkk\Desktop\Bitirme_Projesi\data\processed_dataset_128"
    
    mri_df = pd.read_csv(mri_csv_path)
    existing_paths = []
    for p in mri_df["path"]:
        if os.path.exists(os.path.join(processed_dir, f"{p}.npy")):
            existing_paths.append(p)
    mri_df = mri_df[mri_df["path"].isin(existing_paths)].reset_index(drop=True)
    
    def coerce_label(v):
        try: v = float(v)
        except: v = 0.0
        return 0 if v == 0.0 else (1 if v == 0.5 else 2)
        
    labels = np.array([coerce_label(v) for v in mri_df["memory"]], dtype=np.int64)
    groups = mri_df["OASISID"].astype(str).to_numpy()
    
    from sklearn.model_selection import StratifiedGroupKFold
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_val_idx, test_idx = next(outer.split(mri_df, labels, groups))
    mri_train_val = mri_df.iloc[train_val_idx].reset_index(drop=True)
    mri_test = mri_df.iloc[test_idx].reset_index(drop=True)
    
    inner_labels = np.array([coerce_label(v) for v in mri_train_val["memory"]], dtype=np.int64)
    inner_groups = mri_train_val["OASISID"].astype(str).to_numpy()
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    train_idx, val_idx = next(inner.split(mri_train_val, inner_labels, inner_groups))
    
    mri_train = mri_train_val.iloc[train_idx].reset_index(drop=True)
    mri_val = mri_train_val.iloc[val_idx].reset_index(drop=True)
    
    train_paths = set(mri_train["path"].tolist())
    val_paths = set(mri_val["path"].tolist())
    test_paths = set(mri_test["path"].tolist())
    
    # Filter clinical DF by paths (keep one row per path exactly to match MRI)
    train_df = df[df["path"].isin(train_paths)].drop_duplicates(subset="path", keep="first").reset_index(drop=True)
    val_df = df[df["path"].isin(val_paths)].drop_duplicates(subset="path", keep="first").reset_index(drop=True)
    test_df = df[df["path"].isin(test_paths)].drop_duplicates(subset="path", keep="first").reset_index(drop=True)
    
    return train_df, val_df, test_df

def encode_labels(df: pd.DataFrame) -> np.ndarray:
    label_map = {0: 0, 0.5: 1, 1: 2, 2: 2}
    labels = []
    for v in df["memory"]:
        try: v_f = float(v)
        except: v_f = 0.0
        labels.append(label_map.get(v_f, 0))
    return np.array(labels, dtype=np.int64)

def plot_roc_curve(labels: np.ndarray, probs: np.ndarray, title: str, filename: str) -> None:
    y_bin = label_binarize(labels, classes=[0, 1, 2])
    n_classes = y_bin.shape[1]
    
    fpr, tpr, roc_auc = dict(), dict(), dict()
    colors = ['#2196F3', '#FF9800', '#E91E63'] 
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure(figsize=(8, 6))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{CFG["class_names"][i]} (AUC = {roc_auc[i]:0.3f})')
    
    plt.plot(fpr["micro"], tpr["micro"], color='black', lw=2, linestyle='--', label=f'Micro-avg (AUC = {roc_auc["micro"]:0.3f})')
    plt.plot([0, 1], [0, 1], 'k:', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def print_and_plot_metrics(y_true, y_pred, y_prob, prefix: str, title_prefix: str):
    acc = accuracy_score(y_true, y_pred) * 100.0
    bacc = balanced_accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro") * 100.0
    
    print(f"\n============================================================")
    print(f"{title_prefix.upper()} METRİKLERİ")
    print(f"============================================================")
    print(f"Accuracy: {acc:.2f}% | Balanced Accuracy: {bacc:.2f}% | Macro-F1: {macro_f1:.2f}%")
    print(classification_report(y_true, y_pred, target_names=CFG["class_names"], zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    
    class_accs = cm.diagonal() / cm.sum(axis=1)
    normal_acc = class_accs[0] * 100.0
    
    plt.figure(figsize=(8, 6))
    labels = [f"{count}\n({pct:.1%})" for count, pct in zip(cm.flatten(), cm_norm.flatten())]
    labels = np.asarray(labels).reshape(3, 3)
    sns.heatmap(cm_norm, annot=labels, fmt="", cmap='Blues', xticklabels=CFG["class_names"], yticklabels=CFG["class_names"])
    plt.title(f"{CFG['model_name']} - {title_prefix} Confusion Matrix\nAccuracy: {acc:.2f}% | BAcc: {bacc:.2f}%")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    cm_path = os.path.join(CFG["graphs_dir"], f"xgboost_{prefix}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✅ Confusion Matrix kaydedildi: {cm_path}")
    
    # ROC Curve
    roc_path = os.path.join(CFG["graphs_dir"], f"xgboost_{prefix}_roc_curve.png")
    plot_roc_curve(y_true, y_prob, f"{title_prefix} ROC Curves (XGBoost)", roc_path)
    print(f"✅ ROC Eğrisi kaydedildi: {roc_path}")

# ============================================================
# 3. MAIN (Test Akışı)
# ============================================================
def main():
    os.makedirs(CFG["graphs_dir"], exist_ok=True)

    # 1. Veri Yükle & Temizle
    data_df = pd.read_csv(CFG["csv_path"])
    
    clinical_cols = [
        col for col in data_df.columns 
        if col not in ['OASISID', 'memory', 'path'] and not str(col).startswith('Unnamed:')
    ]
    
    train_df, val_df, test_df = make_group_splits(data_df)
    
    for col in clinical_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        val_df[col] = pd.to_numeric(val_df[col], errors='coerce').fillna(0)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
    
    X_val = val_df[clinical_cols].values
    y_val = encode_labels(val_df)
    
    X_test = test_df[clinical_cols].values
    y_test = encode_labels(test_df)
    
    print(f"Validation seti boyutu: {len(y_val)}")
    print(f"Test seti boyutu: {len(y_test)}")

    # 2. Modeli Yükle
    model = xgb.XGBClassifier()
    model_path = os.path.join(CFG["weights_dir"], f"{CFG['model_name']}_model.json")
    
    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        return
        
    model.load_model(model_path)
    print(f"Model yüklendi: {model_path}")

    # 3. Validation Değerlendirmesi
    val_probs = model.predict_proba(X_val)
    val_preds = model.predict(X_val)
    print_and_plot_metrics(y_val, val_preds, val_probs, "val", "Validation")

    # 4. Test Değerlendirmesi
    test_probs = model.predict_proba(X_test)
    test_preds = model.predict(X_test)
    print_and_plot_metrics(y_test, test_preds, test_probs, "test", "Test")

    # 5. Feature Importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALİZİ")
    print("="*60)
    
    importances = model.feature_importances_
    df_importance = pd.DataFrame({
        'Feature': clinical_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    imp_csv_path = os.path.join(CFG["graphs_dir"], "xgboost_feature_importance.csv")
    df_importance.to_csv(imp_csv_path, index=False)
    print(f"✅ Feature Importance CSV kaydedildi: {imp_csv_path}")
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=df_importance.head(30), palette='viridis')
    plt.title(f'Top 30 Important Features (XGBoost)\nBest Composite Score Optuna Parameters')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    imp_plot_path = os.path.join(CFG["graphs_dir"], "xgboost_feature_importance.png")
    plt.savefig(imp_plot_path, dpi=150)
    plt.close()
    print(f"✅ Feature Importance grafiği kaydedildi: {imp_plot_path}")

if __name__ == "__main__":
    main()
