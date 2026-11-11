
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# Ayarlar
INPUT_FILE = "scripts/7-embedded_meds1.csv"
OUTPUT_FILE = "scripts/7-embedded_meds_pca_8.csv" # 8 Boyuta indir (veya 4)
TARGET_DIM = 8

def reduce_dimensions():
    if not os.path.exists(INPUT_FILE):
        print(f"Hata: {INPUT_FILE} bulunamadı.")
        return

    # Veriyi oku
    df = pd.read_csv(INPUT_FILE)
    print(f"Orijinal veri boyutu: {df.shape}")
    
    # İlaç isimlerini ayır
    med_names = df['medicine']
    embeddings = df.drop('medicine', axis=1).values
    
    print(f"Embedding martix shape: {embeddings.shape}")
    
    # PCA Uygula
    pca = PCA(n_components=TARGET_DIM)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Explained Variance (Total): {explained_variance:.4f}")
    print(f"Explained Variance Ratio per component: {pca.explained_variance_ratio_}")
    
    if explained_variance < 0.90:
        print("UYARI: Varyansın %90'ından azı korunuyor. Boyutu artırmayı düşünün.")
    else:
        print("✅ Varyansın büyük kısmı korunuyor.")

    # Yeni DataFrame oluştur
    cols = [f"pca_{i}" for i in range(TARGET_DIM)]
    df_reduced = pd.DataFrame(reduced_embeddings, columns=cols)
    df_reduced.insert(0, "medicine", med_names)
    
    # Kaydet
    df_reduced.to_csv(OUTPUT_FILE, index=False)
    print(f"İndirgenmiş veri kaydedildi: {OUTPUT_FILE}")
    print(df_reduced.head())
    
    # Scree Plot (Ek bilgi)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, TARGET_DIM + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
    plt.title('Explained Variance by Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid()
    plt.savefig("scripts/embedding_pca_variance.png")
    print("Grafik kaydedildi: scripts/embedding_pca_variance.png")

if __name__ == "__main__":
    reduce_dimensions()
