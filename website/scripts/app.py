import streamlit as st
import numpy as np
import pandas as pd
import torch

# utils.py'den fonksiyonları import et
from utils import (
    CNN_MLP_Fusion, CLASS_NAMES, NUM_CLASSES, CLINICAL_DIM,
    load_model, load_embedding_dataframes, create_scaler_from_training_data,
    build_embeddings_from_names, load_and_preprocess_mr_from_bytes,
    prepare_clinical_data, predict_alzheimer, load_patient_from_json
)

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Alzheimer Tahmin Sistemi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# MODERN CSS
# ============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #0a0a0f;
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding: 2rem 3rem;
    max-width: 1200px;
}

/* ── Subtle Animated Background ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: 
        radial-gradient(ellipse 80% 50% at 20% 40%, rgba(99, 102, 241, 0.08) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 80% 20%, rgba(168, 85, 247, 0.06) 0%, transparent 70%),
        radial-gradient(ellipse 50% 60% at 50% 80%, rgba(6, 182, 212, 0.04) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 50%, #0f0f1a 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.15);
}

[data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }
[data-testid="stSidebar"] label { color: #cbd5e1 !important; font-weight: 500; font-size: 0.85rem; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label { color: #cbd5e1 !important; }

[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    border: 1px dashed rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 8px;
}

/* ── Hero ── */
.hero-container {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 0.75rem;
    color: #818cf8;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.75rem 0;
    line-height: 1.1;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #64748b;
    font-weight: 400;
    margin: 0;
    max-width: 500px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Metric Cards ── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 2rem 0;
}

.metric-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.metric-card:hover {
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(99, 102, 241, 0.2);
    transform: translateY(-2px);
}

.metric-card:hover::before { opacity: 1; }

.metric-icon {
    font-size: 1.8rem;
    margin-bottom: 12px;
    display: block;
}

.metric-card h4 {
    color: #e2e8f0;
    font-size: 0.95rem;
    font-weight: 600;
    margin: 0 0 6px 0;
}

.metric-card p {
    color: #64748b;
    font-size: 0.8rem;
    margin: 0;
    line-height: 1.4;
}

/* ── CTA Button ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 16px 48px;
    font-size: 1rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.3px;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3),
                inset 0 1px 0 rgba(255,255,255,0.1);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(99, 102, 241, 0.45),
                inset 0 1px 0 rgba(255,255,255,0.15);
}

.stButton > button:active {
    transform: translateY(0);
}

/* ── Result Cards ── */
.results-header {
    text-align: center;
    margin: 2rem 0 1.5rem;
}

.results-header h2 {
    color: #e2e8f0;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
}

.result-card {
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.result-card.inactive-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
}

.result-card.active-cn {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(16, 185, 129, 0.1) 100%);
    border: 1px solid rgba(34, 197, 94, 0.4);
    box-shadow: 0 0 40px rgba(34, 197, 94, 0.1);
}

.result-card.active-mci {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(251, 191, 36, 0.1) 100%);
    border: 1px solid rgba(245, 158, 11, 0.4);
    box-shadow: 0 0 40px rgba(245, 158, 11, 0.1);
}

.result-card.active-ad {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(244, 63, 94, 0.1) 100%);
    border: 1px solid rgba(239, 68, 68, 0.4);
    box-shadow: 0 0 40px rgba(239, 68, 68, 0.1);
}

.result-card:hover {
    transform: translateY(-3px);
}

.result-card h3 {
    margin: 0 0 8px 0;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.result-card .prob-value {
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0;
    line-height: 1;
    letter-spacing: -1px;
}

.inactive-card h3 { color: #475569; }
.inactive-card .prob-value { color: #334155; }

.active-cn h3 { color: #86efac; }
.active-cn .prob-value { color: #22c55e; }

.active-mci h3 { color: #fcd34d; }
.active-mci .prob-value { color: #f59e0b; }

.active-ad h3 { color: #fca5a5; }
.active-ad .prob-value { color: #ef4444; }

/* ── Diagnosis Panel ── */
.diagnosis-panel {
    border-radius: 16px;
    padding: 28px;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}

.diagnosis-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 4px;
}

.diagnosis-cn {
    background: rgba(34, 197, 94, 0.06);
    border: 1px solid rgba(34, 197, 94, 0.15);
}
.diagnosis-cn::before { background: #22c55e; }

.diagnosis-mci {
    background: rgba(245, 158, 11, 0.06);
    border: 1px solid rgba(245, 158, 11, 0.15);
}
.diagnosis-mci::before { background: #f59e0b; }

.diagnosis-ad {
    background: rgba(239, 68, 68, 0.06);
    border: 1px solid rgba(239, 68, 68, 0.15);
}
.diagnosis-ad::before { background: #ef4444; }

.diagnosis-panel h3 {
    font-size: 1.15rem;
    font-weight: 700;
    margin: 0 0 8px 0;
}

.diagnosis-cn h3 { color: #22c55e; }
.diagnosis-mci h3 { color: #f59e0b; }
.diagnosis-ad h3 { color: #ef4444; }

.diagnosis-panel .diag-meta {
    font-size: 0.9rem;
    color: #94a3b8;
    margin: 0 0 12px 0;
}

.diagnosis-panel .diag-msg {
    color: #cbd5e1;
    font-size: 0.9rem;
    line-height: 1.6;
    margin: 0;
}

/* ── MR Viewer Card ── */
.mr-viewer {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 24px;
    margin-top: 1.5rem;
}

.mr-viewer-title {
    color: #e2e8f0;
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 16px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 2rem 1rem;
    margin-top: 3rem;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
}

.app-footer .footer-warning {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.15);
    border-radius: 10px;
    padding: 10px 18px;
    font-size: 0.8rem;
    color: #fbbf24;
    margin-bottom: 16px;
}

.app-footer .footer-info {
    color: #475569;
    font-size: 0.75rem;
    margin: 4px 0;
}

/* ── Sidebar Section Headers ── */
.sidebar-section { 
    color: #818cf8;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 1rem 0 0.5rem;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(99, 102, 241, 0.15);
}

/* ── Expander Styling ── */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
}

[data-testid="stExpander"] summary {
    color: #94a3b8;
    font-size: 0.85rem;
}

/* ── Divider ── */
hr {
    border: none;
    height: 1px;
    background: rgba(255, 255, 255, 0.05);
    margin: 1rem 0;
}

/* ── Hide Streamlit Defaults ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Loading spinner ── */
.stSpinner > div { border-color: #6366f1 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================
# CACHE'Lİ VERİ YÜKLEYİCİLER
# ============================================
@st.cache_resource
def cached_load_model():
    return load_model("../weights/best_model.pth")

@st.cache_data
def cached_load_embeddings():
    return load_embedding_dataframes("../repository/embedded_meds1.csv", "../repository/embedded_dx1.csv")

@st.cache_data
def cached_load_scaler():
    return create_scaler_from_training_data("../repository/data.csv")

@st.cache_data
def get_medicine_list():
    meds_df = pd.read_csv("../repository/embedded_meds1.csv")
    return [""] + sorted(meds_df['medicine'].dropna().unique().tolist())

@st.cache_data
def get_diagnosis_list():
    dx_df = pd.read_csv("../repository/embedded_dx1.csv")
    return [""] + sorted(dx_df['dx'].dropna().unique().tolist())

@st.cache_data
def load_default_patient():
    try:
        return load_patient_from_json("../repository/patient_data.json")
    except:
        return None

# ============================================
# HERO SECTION
# ============================================
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🧠 Yapay Zeka Destekli</div>
    <h1 class="hero-title">Alzheimer Tahmin<br>Sistemi</h1>
    <p class="hero-subtitle">3D MRI görüntüleri ve klinik verileri analiz ederek erken teşhis ve risk değerlendirmesi yapan hibrit derin öğrenme modeli</p>
</div>
""", unsafe_allow_html=True)

# Feature Metrics
st.markdown("""
<div class="metrics-grid">
    <div class="metric-card" style="--accent: #6366f1;">
        <span class="metric-icon">🧬</span>
        <h4>3D CNN Analizi</h4>
        <p>Custom ResNet + SE bloklarla<br>volumetrik MRI analizi</p>
    </div>
    <div class="metric-card" style="--accent: #8b5cf6;">
        <span class="metric-icon">📊</span>
        <h4>Klinik Veri Füzyonu</h4>
        <p>Cross-Modal Attention ile<br>çoklu veri kaynağı birleştirme</p>
    </div>
    <div class="metric-card" style="--accent: #a855f7;">
        <span class="metric-icon">⚡</span>
        <h4>Hibrit Model</h4>
        <p>CNN + MLP Fusion mimarisi<br>ile yüksek doğruluk</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# VERİ YÜKLEME
# ============================================
try:
    model, device = cached_load_model()
    meds_df, dx_df = cached_load_embeddings()
    scaler, clinical_cols, mean_dict = cached_load_scaler()
    medicine_list = get_medicine_list()
    diagnosis_list = get_diagnosis_list()
    default_patient = load_default_patient()
    st.sidebar.success("✓ Model ve veriler yüklendi")
except Exception as e:
    st.error(f"Veri yükleme hatası: {e}")
    st.stop()

# Varsayılan değerleri ayarla
if default_patient:
    default_meds = default_patient.get('medicines', {})
    default_dx = default_patient.get('diagnoses', {})
    default_clinical = default_patient.get('clinical_data', {})
else:
    default_meds = {}
    default_dx = {}
    default_clinical = {}

# ============================================
# SIDEBAR - GİRİŞ FORMU
# ============================================
st.sidebar.markdown('<div class="sidebar-section">📋 Hasta Bilgileri</div>', unsafe_allow_html=True)

# MR Dosyası
st.sidebar.markdown("---")
st.sidebar.markdown("**🧲 MR Görüntüsü**")
mr_file = st.sidebar.file_uploader(
    "MR dosyası yükleyin", 
    type=["npy", "nii", "gz"],
    help="Desteklenen: .nii / .nii.gz (NIfTI) • .npy (numpy)"
)

# İlaçlar
st.sidebar.markdown("---")
st.sidebar.markdown("**💊 Kullanılan İlaçlar**")

def get_default_index(item_list, value):
    if value and value in item_list:
        return item_list.index(value)
    return 0

col1, col2 = st.sidebar.columns(2)
with col1:
    med1 = st.selectbox("İlaç 1", medicine_list, index=get_default_index(medicine_list, default_meds.get('med1')), key="med1")
    med3 = st.selectbox("İlaç 3", medicine_list, index=get_default_index(medicine_list, default_meds.get('med3')), key="med3")
with col2:
    med2 = st.selectbox("İlaç 2", medicine_list, index=get_default_index(medicine_list, default_meds.get('med2')), key="med2")
    med4 = st.selectbox("İlaç 4", medicine_list, index=get_default_index(medicine_list, default_meds.get('med4')), key="med4")

# Tanılar
st.sidebar.markdown("---")
st.sidebar.markdown("**🩺 Mevcut Tanılar**")
dx1 = st.sidebar.selectbox("Tanı 1", diagnosis_list, index=get_default_index(diagnosis_list, default_dx.get('dx1')), key="dx1")
dx2 = st.sidebar.selectbox("Tanı 2", diagnosis_list, index=get_default_index(diagnosis_list, default_dx.get('dx2')), key="dx2")
dx3 = st.sidebar.selectbox("Tanı 3", diagnosis_list, index=get_default_index(diagnosis_list, default_dx.get('dx3')), key="dx3")
dx4 = st.sidebar.selectbox("Tanı 4", diagnosis_list, index=get_default_index(diagnosis_list, default_dx.get('dx4')), key="dx4")
dx5 = st.sidebar.selectbox("Tanı 5", diagnosis_list, index=get_default_index(diagnosis_list, default_dx.get('dx5')), key="dx5")

# Klinik Bilgiler
st.sidebar.markdown("---")
st.sidebar.markdown("**📈 Klinik Değerlendirme**")

embedding_cols = [col for col in clinical_cols if 'emb_' in col]
non_embedding_cols = [col for col in clinical_cols if 'emb_' not in col]

user_inputs = {}

important_cols = ['age at visit', 'MMSE', 'naccudsd', 'cdr', 'gds', 'GDS']
for col in important_cols:
    if col in non_embedding_cols:
        default_val = default_clinical.get(col, mean_dict.get(col, 0))
        if col == 'age at visit':
            user_inputs[col] = st.sidebar.number_input("Yaş", min_value=40, max_value=120, value=int(default_val))
        elif col == 'MMSE':
            user_inputs[col] = st.sidebar.slider("MMSE Skoru", 0, 30, int(default_val), help="Mini Mental State Examination")
        elif col == 'cdr':
            user_inputs[col] = st.sidebar.slider("CDR Skoru", 0.0, 3.0, float(default_val), step=0.5, help="Clinical Dementia Rating")
        elif col in ('gds', 'GDS'):
            user_inputs[col] = st.sidebar.slider("GDS Skoru", 0, 15, int(default_val), help="Geriatric Depression Scale")
        else:
            user_inputs[col] = st.sidebar.number_input(col, value=float(default_val))

with st.sidebar.expander("⚙️ Diğer Klinik Veriler"):
    for col in non_embedding_cols:
        if col not in important_cols:
            default_val = default_clinical.get(col, mean_dict.get(col, 0))
            user_inputs[col] = st.number_input(col, value=float(default_val), key=f"input_{col}")

# ============================================
# TAHMİN BUTONU
# ============================================
st.markdown("<br>", unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    predict_button = st.button("🔬  Analiz Başlat", type="primary", use_container_width=True)

if predict_button:
    if mr_file is None:
        st.error("⚠️ Lütfen bir MR dosyası yükleyin!")
    else:
        try:
            with st.spinner("Model analiz yapıyor..."):
                # 1. MR görüntüsünü işle
                mr_data = load_and_preprocess_mr_from_bytes(mr_file.read(), mr_file.name)
                
                # 2. Embedding değerlerini oluştur
                embedding_values = build_embeddings_from_names(
                    meds_df, dx_df,
                    med1=med1, med2=med2, med3=med3, med4=med4,
                    dx1=dx1, dx2=dx2, dx3=dx3, dx4=dx4, dx5=dx5
                )
                
                # 3. Tüm klinik verileri birleştir
                clinical_dict = {}
                clinical_dict.update(user_inputs)
                clinical_dict.update(embedding_values)
                
                for col in clinical_cols:
                    if col not in clinical_dict:
                        clinical_dict[col] = mean_dict.get(col, 0)
                
                # 4. Tahmin yap
                prediction, probabilities, class_name = predict_alzheimer(
                    mr_data, clinical_dict, model, device, clinical_cols, scaler
                )
            
            # ============================================
            # SONUÇLAR
            # ============================================
            st.markdown("""
            <div class="results-header">
                <h2>📋 Analiz Sonuçları</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Olasılık kartları
            col1, col2, col3 = st.columns(3)
            
            with col1:
                card_class = "active-cn" if prediction == 0 else "inactive-card"
                st.markdown(f"""
                <div class="result-card {card_class}">
                    <h3>CN — Normal</h3>
                    <p class="prob-value">{probabilities[0]*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                card_class = "active-mci" if prediction == 1 else "inactive-card"
                st.markdown(f"""
                <div class="result-card {card_class}">
                    <h3>MCI — Hafif Bozukluk</h3>
                    <p class="prob-value">{probabilities[1]*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                card_class = "active-ad" if prediction == 2 else "inactive-card"
                st.markdown(f"""
                <div class="result-card {card_class}">
                    <h3>AD — Alzheimer</h3>
                    <p class="prob-value">{probabilities[2]*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Tanı paneli
            if prediction == 0:
                panel_class = "diagnosis-cn"
                title = "Düşük Risk — Normal Bilişsel Fonksiyon"
                message = "Mevcut bilişsel fonksiyonlarınız normal sınırlar içindedir. Sağlıklı yaşam tarzınızı sürdürün ve düzenli kontrollerinizi aksatmayın."
            elif prediction == 1:
                panel_class = "diagnosis-mci"
                title = "Orta Risk — Hafif Bilişsel Bozukluk"
                message = "Hafif bilişsel bozukluk belirtileri tespit edilmiştir. Bir nörolog ile görüşmenizi ve bilişsel egzersizlere ağırlık vermenizi öneririz."
            else:
                panel_class = "diagnosis-ad"
                title = "Yüksek Risk — Alzheimer Hastalığı"
                message = "Alzheimer hastalığı belirtileri tespit edilmiştir. Acil olarak bir nörolog ile görüşün ve detaylı değerlendirme yaptırın."
            
            st.markdown(f"""
            <div class="diagnosis-panel {panel_class}">
                <h3>{title}</h3>
                <p class="diag-meta"><strong>Tahmin:</strong> {class_name}  ·  <strong>Güven:</strong> {probabilities[prediction]*100:.1f}%</p>
                <p class="diag-msg">{message}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # MR görüntüsünü göster
            st.markdown("""
            <div class="mr-viewer">
                <p class="mr-viewer-title">🧲 MR Görüntü Kesitleri</p>
            </div>
            """, unsafe_allow_html=True)
            
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.patch.set_facecolor('#0a0a0f')
            
            mid_x = mr_data.shape[0] // 2
            mid_y = mr_data.shape[1] // 2
            mid_z = mr_data.shape[2] // 2
            
            for ax, data_slice, title in [
                (axes[0], mr_data[mid_x, :, :], 'Sagittal'),
                (axes[1], mr_data[:, mid_y, :], 'Koronal'),
                (axes[2], mr_data[:, :, mid_z], 'Aksiyel')
            ]:
                ax.imshow(data_slice, cmap='gray')
                ax.set_title(title, fontsize=13, fontweight='600', color='#e2e8f0', pad=10)
                ax.axis('off')
                ax.set_facecolor('#0a0a0f')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
        except Exception as e:
            st.error(f"Tahmin hatası: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="app-footer">
    <div class="footer-warning">⚠️ Bu uygulama sadece bilgilendirme amaçlıdır. Kesin tanı için bir sağlık uzmanına başvurun.</div>
    <p class="footer-info">Alzheimer Tahmin Sistemi · CNN + MLP Fusion Hibrit Model · <strong>Bitirme Projesi 2025</strong></p>
    <p class="footer-info" style="opacity: 0.5;">Powered by PyTorch & Streamlit</p>
</div>
""", unsafe_allow_html=True)
