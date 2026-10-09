"""Transfer Learning Classification Module for AquaVision.

Implements deep visual feature extraction using pretrained ImageNet CNN backbones
(EfficientNet-B0, MobileNetV3-Small) with calibrated classification heads for fish species identification.
"""

from __future__ import annotations

import time
from typing import Literal
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torchvision import models, transforms


def validate_image(image: Image.Image) -> None:
    """Validate image format, type, and minimum dimensions."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(image).__name__}")
    if image.width < 16 or image.height < 16:
        raise ValueError(
            f"Image dimensions ({image.width}x{image.height}) are too small. Minimum supported size is 16x16."
        )


class TransferLearningClassifier:
    """Pretrained deep CNN feature extractor coupled with a calibrated linear classification head."""

    temperature: float = 2.2819

    def __init__(
        self,
        backbone_name: Literal["efficientnet_b0", "mobilenet_v3_small"] = "efficientnet_b0",
        random_state: int = 42,
        temperature: float = 2.2819,
    ) -> None:
        self.backbone_name = backbone_name
        self.random_state = random_state
        self.classes_: np.ndarray = np.array([])
        self.scaler = StandardScaler()
        self.head = LogisticRegression(max_iter=1000, random_state=random_state)
        self.feature_dim = 1280 if backbone_name == "efficientnet_b0" else 576
        self.temperature = float(temperature)

        self._preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self._init_backbone()

    def _init_backbone(self) -> None:
        if self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
            self._extractor = nn.Sequential(model.features, model.avgpool, nn.Flatten())
        elif self.backbone_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            model = models.mobilenet_v3_small(weights=weights)
            self._extractor = nn.Sequential(model.features, model.avgpool, nn.Flatten())
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        self._extractor.eval()
        for param in self._extractor.parameters():
            param.requires_grad = False

    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Extract deep visual feature embedding from a single PIL image."""
        validate_image(image)
        rgb_img = image.convert("RGB")
        tensor = self._preprocess(rgb_img).unsqueeze(0)
        with torch.no_grad():
            feat = self._extractor(tensor).squeeze(0).cpu().numpy()
        return feat.astype(np.float32)

    def extract_features_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Extract deep embeddings in batches for efficiency."""
        feats: list[np.ndarray] = []
        for img in images:
            feats.append(self.extract_features(img))
        return np.vstack(feats)

    def fit(self, X: np.ndarray, y: np.ndarray) -> TransferLearningClassifier:
        """Fit the classification head on extracted deep features."""
        self.classes_ = np.array(sorted(list(set(y))))
        X_scaled = self.scaler.fit_transform(X)
        self.head.fit(X_scaled, y)
        return self

    def predict_proba(self, X: np.ndarray, apply_temperature: bool = True) -> np.ndarray:
        """Predict class probabilities from feature embeddings with temperature scaling calibration.
        
        Preserves an automatic safe fallback to uncalibrated raw softmax if temperature scaling
        is unavailable or temperature is set to 1.0.
        """
        X_scaled = self.scaler.transform(X)
        temp = getattr(self, "temperature", 1.0)
        if apply_temperature and temp != 1.0 and temp > 0.0 and hasattr(self.head, "decision_function"):
            try:
                logits = self.head.decision_function(X_scaled)
                scaled = logits / temp
                exp_z = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
                return exp_z / np.sum(exp_z, axis=1, keepdims=True)
            except Exception:
                pass
        return self.head.predict_proba(X_scaled)

    def predict_proba_image(self, image: Image.Image) -> np.ndarray:
        """Extract features and predict class probabilities directly from PIL image."""
        feat = self.extract_features(image).reshape(1, -1)
        return self.predict_proba(feat)[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict species classes from feature embeddings."""
        X_scaled = self.scaler.transform(X)
        return self.head.predict(X_scaled)

    def predict_image(self, image: Image.Image) -> str:
        """Predict species class directly from PIL image."""
        probs = self.predict_proba_image(image)
        best_idx = int(np.argmax(probs))
        return str(self.classes_[best_idx])
