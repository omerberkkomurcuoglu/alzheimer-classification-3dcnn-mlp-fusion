"""
Alzheimer Classification API
=============================
FastAPI backend serving ResNet50 + XGBoost late-fusion predictions.

Endpoints:
    POST /api/predict           — Full fusion (clinical JSON + MRI file)
    POST /api/predict/clinical  — Clinical-only prediction
    GET  /api/default-patient   — Default severe-patient data for testing
    GET  /api/health            — Health check
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import (
    CLASS_NAMES,
    CLINICAL_FEATURE_COLS,
    DEFAULT_TEST_MRI_PATH,
    RESNET50_WEIGHTS,
    XGBOOST_WEIGHTS,
)
from .models import PredictionResponse
from .services.clinical_service import ClinicalService
from .services.fusion_service import FusionService
from .services.mri_service import MRIService

# ---------------------------------------------------------------------------
# Service instances (lazy-loaded)
# ---------------------------------------------------------------------------
mri_service = MRIService(RESNET50_WEIGHTS)
clinical_service = ClinicalService(XGBOOST_WEIGHTS)
fusion_service = FusionService()

# Default patient JSON path
DEFAULT_PATIENT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "inference", "test_data", "default_patient.json"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load models on startup for faster first request."""
    try:
        clinical_service.load()
        print("[Startup] Clinical model ready.")
    except Exception as e:
        print(f"[Startup] Clinical model load deferred: {e}")
    # MRI model is large — load lazily on first request
    yield


app = FastAPI(
    title="Alzheimer Classification API",
    version="1.0.0",
    description="ResNet50 + XGBoost Late Fusion for Alzheimer's Disease Classification",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "models": {"clinical": clinical_service._loaded, "mri": mri_service._loaded}}


@app.get("/api/default-patient")
async def default_patient():
    """Return the default severe-patient data for one-click demo."""
    try:
        with open(DEFAULT_PATIENT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "Default patient file not found."})


@app.post("/api/predict/clinical")
async def predict_clinical(clinical_json: str = Form(...)):
    """Clinical-only prediction (no MRI)."""
    try:
        clinical_dict = json.loads(clinical_json)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON."})

    clinical_probs = clinical_service.predict(clinical_dict)
    result = fusion_service.fuse(mri_probs=None, clinical_probs=clinical_probs)
    return JSONResponse(content=result)


@app.post("/api/predict")
async def predict_fusion(
    clinical_json: str = Form(...),
    mri_file: Optional[UploadFile] = File(None),
    use_default_mri: bool = Form(False),
):
    """
    Full fusion prediction.

    Parameters
    ----------
    clinical_json : str (Form)
        JSON string of clinical features.
    mri_file : UploadFile (optional)
        Uploaded .npy or .nii.gz MRI volume.
    use_default_mri : bool (Form, default=False)
        If True and no mri_file, use the default test MRI.
    """
    # Parse clinical data
    try:
        clinical_dict = json.loads(clinical_json)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid clinical JSON."})

    # Clinical inference
    clinical_probs = clinical_service.predict(clinical_dict)

    # MRI inference
    mri_probs = None
    if mri_file is not None:
        file_bytes = await mri_file.read()
        mri_probs = mri_service.predict_from_bytes(file_bytes, mri_file.filename)
    elif use_default_mri and os.path.exists(DEFAULT_TEST_MRI_PATH):
        mri_probs = mri_service.predict_from_npy(DEFAULT_TEST_MRI_PATH)

    # Fusion
    result = fusion_service.fuse(mri_probs=mri_probs, clinical_probs=clinical_probs)
    return JSONResponse(content=result)


@app.get("/api/feature-info")
async def feature_info():
    """Return list of expected clinical feature columns."""
    return {"features": CLINICAL_FEATURE_COLS, "class_names": CLASS_NAMES}
