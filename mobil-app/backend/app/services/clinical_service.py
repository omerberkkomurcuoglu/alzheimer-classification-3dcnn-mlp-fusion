"""
Clinical Inference Service — XGBoost for clinical tabular data.
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from ..config import CLINICAL_FEATURE_COLS


class ClinicalService:
    """Load XGBoost model and predict from clinical feature dict."""

    def __init__(self, weights_path: str):
        self.weights_path = weights_path
        self.model = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        print(f"[Clinical] Loading XGBoost from {self.weights_path} ...")
        self.model = xgb.XGBClassifier()
        self.model.load_model(self.weights_path)
        self._loaded = True
        print("[Clinical] XGBoost loaded ✅")

    def _build_feature_vector(self, clinical_dict: dict) -> np.ndarray:
        """Convert a flat dict of clinical values into a feature vector
        following the exact column order used during training."""
        features = []
        for col in CLINICAL_FEATURE_COLS:
            val = clinical_dict.get(col, 0.0)
            try:
                features.append(float(val))
            except (TypeError, ValueError):
                features.append(0.0)
        return np.array(features, dtype=np.float32).reshape(1, -1)

    def predict(self, clinical_dict: dict) -> np.ndarray:
        """Return [3] class probabilities from clinical features."""
        self.load()
        X = self._build_feature_vector(clinical_dict)
        probs = self.model.predict_proba(X)
        return probs[0]  # shape (3,)
