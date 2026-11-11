"""
Alzheimer Classification API - Configuration
"""
import os

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TRAIN_DIR = os.path.join(PROJECT_ROOT, "scripts", "train")
FUSION_WEIGHTS_DIR = os.path.join(TRAIN_DIR, "fusion", "weights")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed_dataset_128")

# Model weights
RESNET50_WEIGHTS = os.path.join(FUSION_WEIGHTS_DIR, "resnet.pth")
XGBOOST_WEIGHTS = os.path.join(FUSION_WEIGHTS_DIR, "xgboost.json")

# Model hyperparameters
RESNET50_DROPOUT = 0.30
NUM_CLASSES = 3
CLASS_NAMES = ["Normal", "Mild", "Moderate/Severe"]
CLASS_DESCRIPTIONS = [
    "Bilişsel fonksiyonlar normal sınırlarda.",
    "Hafif bilişsel bozulma tespit edildi. Takip önerilir.",
    "Orta/Ağır düzeyde bilişsel bozulma. Uzman değerlendirmesi gereklidir.",
]

# Fusion parameters
ALPHA = 0.35  # MRI weight (Clinical = 1 - alpha = 0.65)

# Clinical feature columns (from CSV, excluding OASISID, memory, path, Unnamed)
CLINICAL_FEATURE_COLS = [
    "age at visit", "ANIMALS", "asscmem", "srtfree", "srttotal", "tma", "tmb",
    "INDEPEND", "RESIDENC", "MARISTAT", "INSEX", "INRELTO", "INLIVWTH", "INRELY",
    "anymeds", "WEIGHT", "BPSYS", "BPDIAS", "NPIQINF",
    "DEL", "HALL", "AGIT", "DEPD", "ANX", "ELAT", "APA", "DISN", "IRR", "MOT", "NITE", "APP",
    "SATIS", "DROPACT", "EMPTY", "BORED", "SPIRITS", "AFRAID", "HAPPY", "HELPLESS",
    "STAYHOME", "WONDRFUL", "WRTHLESS", "ENERGY", "HOPELESS", "BETTER", "GDS",
    "BILLS", "TAXES", "SHOPPING", "GAMES", "STOVE", "MEALPREP", "EVENTS", "PAYATTN",
    "REMDATES", "TRAVEL",
    "PSP", "CORT", "HUNT", "PRION", "MEDS", "DYSILL", "DEP", "OTHPSY", "DOWNS",
    "PARK", "HYCEPH", "BRNINJ", "NEOP", "COGOTH", "COGOTH2", "COGOTH3",
    # Medication embeddings (4 meds x 8 dims = 32)
    "meds_1_emb_0", "meds_1_emb_1", "meds_1_emb_2", "meds_1_emb_3",
    "meds_1_emb_4", "meds_1_emb_5", "meds_1_emb_6", "meds_1_emb_7",
    "meds_2_emb_0", "meds_2_emb_1", "meds_2_emb_2", "meds_2_emb_3",
    "meds_2_emb_4", "meds_2_emb_5", "meds_2_emb_6", "meds_2_emb_7",
    "meds_3_emb_0", "meds_3_emb_1", "meds_3_emb_2", "meds_3_emb_3",
    "meds_3_emb_4", "meds_3_emb_5", "meds_3_emb_6", "meds_3_emb_7",
    "meds_4_emb_0", "meds_4_emb_1", "meds_4_emb_2", "meds_4_emb_3",
    "meds_4_emb_4", "meds_4_emb_5", "meds_4_emb_6", "meds_4_emb_7",
]

# Default test patient (OAS30002 - memory=2.0, Moderate/Severe)
DEFAULT_TEST_MRI_PATH = os.path.join(PROCESSED_DATA_DIR, "OAS30002_MR_d0653.npy")
