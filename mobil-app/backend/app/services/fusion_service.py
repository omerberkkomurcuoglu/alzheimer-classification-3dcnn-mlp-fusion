"""
Late Fusion Service — combines MRI and Clinical predictions.
"""
from __future__ import annotations

import numpy as np

from ..config import ALPHA, CLASS_DESCRIPTIONS, CLASS_NAMES


class FusionService:
    """Weighted late fusion of MRI and Clinical probability vectors."""

    def __init__(self, alpha: float = ALPHA):
        self.alpha = alpha

    def fuse(
        self,
        mri_probs: np.ndarray | None,
        clinical_probs: np.ndarray | None,
    ) -> dict:
        """
        Perform late fusion and return a structured result dict.

        Parameters
        ----------
        mri_probs : (3,) array or None
        clinical_probs : (3,) array or None

        Returns
        -------
        dict with predicted_class, label, description, probabilities, confidence, mode
        """
        if mri_probs is not None and clinical_probs is not None:
            fused = self.alpha * mri_probs + (1 - self.alpha) * clinical_probs
            mode = "fusion"
        elif clinical_probs is not None:
            fused = clinical_probs
            mode = "clinical_only"
        elif mri_probs is not None:
            fused = mri_probs
            mode = "mri_only"
        else:
            raise ValueError("At least one modality must be provided.")

        predicted_class = int(np.argmax(fused))
        confidence = float(np.max(fused))
        prob_dict = {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, fused)}

        result = {
            "predicted_class": predicted_class,
            "predicted_label": CLASS_NAMES[predicted_class],
            "description": CLASS_DESCRIPTIONS[predicted_class],
            "probabilities": prob_dict,
            "confidence": round(confidence, 4),
            "fusion_alpha": self.alpha,
            "mode": mode,
        }

        if mri_probs is not None:
            result["mri_probabilities"] = {
                name: round(float(p), 4) for name, p in zip(CLASS_NAMES, mri_probs)
            }
        if clinical_probs is not None:
            result["clinical_probabilities"] = {
                name: round(float(p), 4) for name, p in zip(CLASS_NAMES, clinical_probs)
            }

        return result
