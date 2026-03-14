
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

def create_distribution_plots(file_path, output_path):
    print(f"Reading data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    # Check for 'memory' column
    if 'memory' not in df.columns:
        print("Error: 'memory' column not found in dataset.")
        print("Available columns:", df.columns.tolist())
        return

    # Define Class Mapping based on script logic
    # 0 -> Normal
    # 0.5 -> Very Mild
    # 1.0, 2.0, 3.0 -> Mild/Moderate/Severe (Combined as 'Dementia' or kept separate?)
    # The training script maps {0: 0, 0.5: 1, 1: 2, 2: 2}. 
    # This means 3 classes: Normal, Very Mild, and Dementia (Mild+)
    
    def map_class(val):
        try:
            val = float(val)
        except:
            return "Unknown"
            
        if val == 0:
            return "Cognitively Normal (CN)"
        elif val == 0.5:
            return "Very Mild / MCI"
        elif val >= 1.0:
            return "Dementia (Mild to Severe)"
        else:
            return "Unknown"

    df['Class'] = df['memory'].apply(map_class)
    
    # Filter out Unknown if any
    df = df[df['Class'] != "Unknown"]

    # --- NEW: Calculate Statistics ---
    total_rows = len(df)
    unique_patients = df['OASISID'].nunique()
    
    print("-" * 30)
    print(f"Total Data Rows (MRI/Clinical Entries): {total_rows}")
    print(f"Total Unique Patients (OASISID): {unique_patients}")
    
    # Class Distribution Details
    class_counts = df['Class'].value_counts().sort_index()
    print("\n[Class Distribution]")
    for cls, count in class_counts.items():
        ratio = count / total_rows * 100
        print(f" - {cls}: {count} ({ratio:.2f}%)")
        
    # Clinical Columns (Excluding metadata and target)
    print("\n[Clinical Features Used]")
    ignored_cols = ['Unnamed: 0', 'OASISID', 'path', 'memory', 'Class']
    clinical_cols = [c for c in df.columns if c not in ignored_cols]
    print(f"Total Clinical Features: {len(clinical_cols)}")
    print("Features:", ", ".join(clinical_cols[:20]) + "...") # Print first 20 as sample
    print("-" * 30)

    # Count classes
    # class_counts already calculated above
    
    # Create Figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'Class Distribution in Dataset\n(Total Rows: {total_rows} | Unique Patients: {unique_patients})', fontsize=20, fontweight='bold')

    # 1. Bar Plot
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=axes[0], palette="viridis")
    axes[0].set_title('Count per Class', fontsize=15)
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Number of Subjects', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 5, str(v), ha='center', fontsize=11, fontweight='bold')

    # 2. Pie Chart
    axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
                startangle=140, colors=sns.color_palette("viridis", len(class_counts)))
    axes[1].set_title('Percentage Distribution', fontsize=15)

    # Save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    print(f"Graph saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Path to 11-data.csv
    # Assuming script is run from the directory where 'scripts' is a subdirectory or sibling
    # Using absolute path from knowledge of previous steps
    input_csv = r"c:\Users\berkk\Desktop\Bitirme_Projesi\scripts\12-data_leakage.csv"
    output_png = r"c:\Users\berkk\Desktop\Bitirme_Projesi\scripts\class_distribution.png"
    
    create_distribution_plots(input_csv, output_png)
