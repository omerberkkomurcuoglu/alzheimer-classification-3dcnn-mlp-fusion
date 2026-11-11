"""
MRI Inference Service — ResNet50 3D CNN for brain MRI classification.
"""
from __future__ import annotations

import io
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# Add train dir to path for importing custom modules
_TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "train"))
sys.path.insert(0, _TRAIN_DIR)
sys.path.insert(0, os.path.join(_TRAIN_DIR, "mr"))

from mri_pipeline import load_preprocessed_mri_volume, sanitize_mri_volume

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BrainClassifier(nn.Module):
    """ResNet50 3D classifier — identical architecture to fusion_evaluate_all.py."""

    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        from monai.networks.nets import resnet50 as monai_r50

        backbone = monai_r50(
            spatial_dims=3, n_input_channels=1, num_classes=1,
            shortcut_type="B", bias_downsample=False,
        )
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act = backbone.act
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.act(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.head(self.pool(x))


class MRIService:
    """Singleton-style service to load the ResNet50 model and run MRI inference."""

    def __init__(self, weights_path: str, num_classes: int = 3, dropout: float = 0.30):
        self.model = None
        self.weights_path = weights_path
        self.num_classes = num_classes
        self.dropout = dropout
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        print(f"[MRI] Loading ResNet50 from {self.weights_path} ...")
        self.model = BrainClassifier(self.num_classes, self.dropout).to(DEVICE)
        state = torch.load(self.weights_path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(state)
        self.model.eval()
        self._loaded = True
        print("[MRI] ResNet50 loaded ✅")

    @torch.no_grad()
    def predict_from_npy(self, npy_path: str) -> np.ndarray:
        """Predict from a saved .npy volume (128³). Returns [3] softmax probs."""
        self.load()
        vol = load_preprocessed_mri_volume(npy_path)
        tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,128,128,128)
        logits = self.model(tensor)
        return F.softmax(logits, dim=1).cpu().numpy()[0]

    @torch.no_grad()
    def predict_from_bytes(self, file_bytes: bytes, filename: str) -> np.ndarray:
        """Predict from uploaded file bytes (.npy or .nii.gz)."""
        self.load()
        if filename.endswith(".npy"):
            vol = np.load(io.BytesIO(file_bytes)).astype(np.float32)
            vol = sanitize_mri_volume(vol)
        else:
            # NIfTI support
            import nibabel as nib
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=filename, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                img = nib.load(tmp_path)
                vol = np.asarray(img.dataobj, dtype=np.float32)
                # Resize to 128³ if needed
                if vol.shape != (128, 128, 128):
                    from scipy.ndimage import zoom
                    factors = tuple(128 / s for s in vol.shape[:3])
                    vol = zoom(vol[:, :, :], factors, order=1).astype(np.float32)
                vol = sanitize_mri_volume(vol)
            finally:
                os.unlink(tmp_path)

        tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(DEVICE)
        logits = self.model(tensor)
        return F.softmax(logits, dim=1).cpu().numpy()[0]
