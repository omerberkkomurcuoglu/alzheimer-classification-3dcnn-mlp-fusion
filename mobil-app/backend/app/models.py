"""
Pydantic request / response models for the Alzheimer Classification API.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ClinicalData(BaseModel):
    """Clinical features sent from the mobile app."""

    # Demographic
    age_at_visit: float = Field(..., alias="age at visit", description="Yaş")
    ANIMALS: float = Field(0, description="Hayvan isimlendirme testi skoru")
    asscmem: float = Field(0, description="İlişkisel bellek skoru")
    srtfree: float = Field(0, description="Serbest hatırlama skoru")
    srttotal: float = Field(0, description="Toplam hatırlama skoru")
    tma: float = Field(0, description="Trail Making A (saniye)")
    tmb: float = Field(0, description="Trail Making B (saniye)")

    # Living situation
    INDEPEND: float = Field(1, description="Bağımsızlık düzeyi (1=Bağımsız)")
    RESIDENC: float = Field(1, description="Yaşam yeri (1=Kendi evi)")
    MARISTAT: float = Field(1, description="Medeni durum")
    INSEX: float = Field(1, description="Bilgi veren cinsiyet")
    INRELTO: float = Field(1, description="Bilgi veren yakınlık")
    INLIVWTH: float = Field(0, description="Birlikte yaşama")
    INRELY: float = Field(0, description="Bilgi güvenilirliği")
    anymeds: float = Field(1, description="İlaç kullanımı (0/1)")

    # Physical
    WEIGHT: float = Field(0, description="Kilo (lbs)")
    BPSYS: float = Field(0, description="Sistolik kan basıncı")
    BPDIAS: float = Field(0, description="Diastolik kan basıncı")

    # NPI-Q
    NPIQINF: float = Field(1, description="NPI-Q bilgi kaynağı")
    DEL: float = Field(0, description="Sanrılar")
    HALL: float = Field(0, description="Halüsinasyonlar")
    AGIT: float = Field(0, description="Ajitasyon/Agresyon")
    DEPD: float = Field(0, description="Depresyon/Disfori")
    ANX: float = Field(0, description="Anksiyete")
    ELAT: float = Field(0, description="Öfori/Taşkınlık")
    APA: float = Field(0, description="Apati/Kayıtsızlık")
    DISN: float = Field(0, description="Disinhibisyon")
    IRR: float = Field(0, description="İrritabilite")
    MOT: float = Field(0, description="Motor bozukluk")
    NITE: float = Field(0, description="Gece davranışları")
    APP: float = Field(0, description="İştah/Yeme bozuklukları")

    # GDS
    SATIS: float = Field(0, description="Yaşamdan memnuniyet")
    DROPACT: float = Field(0, description="Aktivite bırakma")
    EMPTY: float = Field(0, description="Boşluk hissi")
    BORED: float = Field(0, description="Sıkılma")
    SPIRITS: float = Field(0, description="Moral")
    AFRAID: float = Field(0, description="Korku")
    HAPPY: float = Field(0, description="Mutluluk")
    HELPLESS: float = Field(0, description="Çaresizlik")
    STAYHOME: float = Field(0, description="Evde kalma tercihi")
    WONDRFUL: float = Field(0, description="Harika hissetme")
    WRTHLESS: float = Field(0, description="Değersizlik hissi")
    ENERGY: float = Field(0, description="Enerji düzeyi")
    HOPELESS: float = Field(0, description="Umutsuzluk")
    BETTER: float = Field(0, description="Daha iyi hissetme")
    GDS: float = Field(0, description="Geriatrik depresyon skoru")

    # FAQ
    BILLS: float = Field(0, description="Fatura ödeme")
    TAXES: float = Field(0, description="Vergi işleri")
    SHOPPING: float = Field(0, description="Alışveriş yapma")
    GAMES: float = Field(0, description="Oyun/hobi takip")
    STOVE: float = Field(0, description="Ocak/cihaz kullanımı")
    MEALPREP: float = Field(0, description="Yemek hazırlama")
    EVENTS: float = Field(0, description="Güncel olayları takip")
    PAYATTN: float = Field(0, description="Dikkat sürdürme")
    REMDATES: float = Field(0, description="Tarih hatırlama")
    TRAVEL: float = Field(0, description="Seyahat")

    # Medical history
    PSP: float = Field(0)
    CORT: float = Field(0)
    HUNT: float = Field(0)
    PRION: float = Field(0)
    MEDS: float = Field(0)
    DYSILL: float = Field(0)
    DEP: float = Field(0)
    OTHPSY: float = Field(0)
    DOWNS: float = Field(0)
    PARK: float = Field(0)
    HYCEPH: float = Field(0)
    BRNINJ: float = Field(0)
    NEOP: float = Field(0)
    COGOTH: float = Field(0)
    COGOTH2: float = Field(0)
    COGOTH3: float = Field(0)

    # Medication embeddings (32 values, default 0)
    meds_1_emb_0: float = 0
    meds_1_emb_1: float = 0
    meds_1_emb_2: float = 0
    meds_1_emb_3: float = 0
    meds_1_emb_4: float = 0
    meds_1_emb_5: float = 0
    meds_1_emb_6: float = 0
    meds_1_emb_7: float = 0
    meds_2_emb_0: float = 0
    meds_2_emb_1: float = 0
    meds_2_emb_2: float = 0
    meds_2_emb_3: float = 0
    meds_2_emb_4: float = 0
    meds_2_emb_5: float = 0
    meds_2_emb_6: float = 0
    meds_2_emb_7: float = 0
    meds_3_emb_0: float = 0
    meds_3_emb_1: float = 0
    meds_3_emb_2: float = 0
    meds_3_emb_3: float = 0
    meds_3_emb_4: float = 0
    meds_3_emb_5: float = 0
    meds_3_emb_6: float = 0
    meds_3_emb_7: float = 0
    meds_4_emb_0: float = 0
    meds_4_emb_1: float = 0
    meds_4_emb_2: float = 0
    meds_4_emb_3: float = 0
    meds_4_emb_4: float = 0
    meds_4_emb_5: float = 0
    meds_4_emb_6: float = 0
    meds_4_emb_7: float = 0

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    """Response from the prediction endpoint."""

    predicted_class: int = Field(..., description="Tahmin edilen sınıf (0, 1, 2)")
    predicted_label: str = Field(..., description="Sınıf etiketi")
    description: str = Field(..., description="Sınıf açıklaması")
    probabilities: dict[str, float] = Field(..., description="Her sınıf için olasılık")
    confidence: float = Field(..., description="En yüksek olasılık (güven skoru)")
    mri_probabilities: Optional[dict[str, float]] = Field(None, description="MRI model olasılıkları")
    clinical_probabilities: Optional[dict[str, float]] = Field(None, description="Klinik model olasılıkları")
    fusion_alpha: float = Field(0.35, description="MRI ağırlığı")
    mode: str = Field("fusion", description="Tahmin modu: fusion | clinical_only")


class DefaultPatientResponse(BaseModel):
    """Response containing default patient data for testing."""

    patient_id: str
    clinical_data: dict
    mri_path: str
    expected_class: str
