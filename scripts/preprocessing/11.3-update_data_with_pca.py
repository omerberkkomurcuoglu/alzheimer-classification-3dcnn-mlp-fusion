import pandas as pd
import numpy as np
import os

# File paths
DATA_11_PATH = "scripts/11-data_backup.csv"
DATA_2_PATH = "scripts/2-filled_data.csv"
PCA_EMBEDDINGS_PATH = "scripts/7-embedded_meds_pca_8.csv"
OUTPUT_PATH = "scripts/11-data.csv"

def update_embeddings():
    print("Loading files...")
    df_11 = pd.read_csv(DATA_11_PATH)
    df_2 = pd.read_csv(DATA_2_PATH)
    df_pca = pd.read_csv(PCA_EMBEDDINGS_PATH)

    print(f"Original 11-data shape: {df_11.shape}")

    # Create a dictionary for fast lookup: medicine -> [pca_0, ... pca_7]
    # Ensure medicine names are lower case and stripped for matching
    df_pca['medicine'] = df_pca['medicine'].astype(str).str.lower().str.strip()
    # Check if there are duplicate medicines in pca file?
    if df_pca['medicine'].duplicated().any():
        print("Warning: Duplicate medications found in PCA file. Using the first occurrence.")
        df_pca = df_pca.drop_duplicates(subset=['medicine'])
    
    pca_map = df_pca.set_index('medicine').T.to_dict('list')
    
    # We need to drop the old embedding columns from df_11
    # They are likely named 'meds_1_emb_0' through 'meds_4_emb_15'
    # Actually, we can just drop ALL matching columns and recreate the 0-7 ones.
    cols_to_drop = []
    for i in range(1, 5):
        for j in range(16):
            col_name = f'meds_{i}_emb_{j}'
            if col_name in df_11.columns:
                cols_to_drop.append(col_name)
    
    print(f"Dropping {len(cols_to_drop)} old embedding columns...")
    df_11 = df_11.drop(columns=cols_to_drop)

    # Merge df_11 with df_2 to get medication names
    # We join on OASISID and age at visit. 
    # Verify column names in df_2
    # From inspection: 'OASISID', 'age at visit', 'meds_1', 'meds_2', 'meds_3', 'meds_4'
    merge_cols = ['OASISID', 'age at visit']
    
    # Ensure types match for merge
    df_11['OASISID'] = df_11['OASISID'].astype(str)
    df_2['OASISID'] = df_2['OASISID'].astype(str)
    
    # 'age at visit' might be float in one and int in another or string
    # Let's force to float for safety or check if they are identical
    df_11['age at visit'] = df_11['age at visit'].astype(float)
    df_2['age at visit'] = df_2['age at visit'].astype(float)

    print("Merging data to retrieve medication names...")
    # We utilize a left join to preserve df_11 rows
    df_merged = pd.merge(
        df_11, 
        df_2[merge_cols + ['meds_1', 'meds_2', 'meds_3', 'meds_4']], 
        on=merge_cols, 
        how='left'
    )
    
    if df_merged.shape[0] != df_11.shape[0]:
        print(f"Warning: Merge changed row count. Original: {df_11.shape[0]}, Merged: {df_merged.shape[0]}")
        # This might happen if duplicates exist in df_2 for the same OASISID+age.
        # Ideally shouldn't happen if df_2 is the source.
    
    # Now generate new columns
    print("Generating new PCA embeddings...")
    
    # Initialize new columns with zeros
    new_embedding_data = {}
    
    # Optimized approach:
    # 1. Map names to whole vectors (resulting in a Series of lists/arrays)
    # 2. Stack them into a DataFrame
 

    # Faster vectorized approach using pandas map
    for med_idx in range(1, 5):
        med_col_name = f'meds_{med_idx}'
        print(f"Processing {med_col_name}...")
        
        names_series = df_merged[med_col_name].astype(str).str.lower().str.strip()
        
        # We want to map each name to a list of 8 floats.
        # If not found, return [0.0]*8
        
        def dict_lookup(x):
            return pca_map.get(x, [0.0]*8)
            
        # Result is a Series of lists
        embeddings_series = names_series.map(dict_lookup)
        
        # Convert Series of lists to DataFrame
        emb_df = pd.DataFrame(embeddings_series.tolist(), index=df_merged.index)
        
        # Rename columns
        emb_df.columns = [f'meds_{med_idx}_emb_{dim}' for dim in range(8)]
        
        # Concatenate to result (we can just add to df_merged directly)
        df_merged = pd.concat([df_merged, emb_df], axis=1)

    # Drop the temporary med name columns
    df_merged = df_merged.drop(columns=['meds_1', 'meds_2', 'meds_3', 'meds_4'])
    
    # Ensure 'path' is the last column if it exists
    if 'path' in df_merged.columns:
        print("Moving 'path' to the last column...")
        cols = [c for c in df_merged.columns if c != 'path'] + ['path']
        df_merged = df_merged[cols]
    
    # Save
    print(f"Saving updated data to {OUTPUT_PATH}...")
    print(f"New shape: {df_merged.shape}")
    df_merged.to_csv(OUTPUT_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    update_embeddings()
