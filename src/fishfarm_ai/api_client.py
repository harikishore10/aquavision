from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import requests
from PIL import Image

from .services import BiomassEstimationResult, CombinedPredictionResult, SpeciesIdentificationResult


@dataclass
class FarmApiClient:
    base_url: str
    token: str
    timeout_seconds: float = 20.0

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _image_payload(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def login(self, username: str, password: str) -> str:
        response = requests.post(
            f"{self.base_url.rstrip('/')}/auth/login",
            json={"username": username, "password": password},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["access_token"]

    def predict_species(self, image: Image.Image) -> SpeciesIdentificationResult:
        response = requests.post(
            f"{self.base_url.rstrip('/')}/species/predict",
            json={"image_base64": self._image_payload(image)},
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return SpeciesIdentificationResult(
            species=payload["species"],
            confidence=payload["confidence"],
            scores=payload["scores"],
            is_reliable=payload.get("is_reliable", True),
            uncertainty_status=payload.get("uncertainty_status", "CONFIDENT"),
            margin_to_second=payload.get("margin_to_second", 0.0),
            entropy=payload.get("entropy", 0.0),
            closed_world_caveat=payload.get("closed_world_caveat", ""),
            warning=payload.get("warning"),
            top_alternatives=payload.get("top_alternatives", []),
        )

    def predict_biomass(self, species: str, length_cm: float, girth_cm: float | None) -> BiomassEstimationResult:
        response = requests.post(
            f"{self.base_url.rstrip('/')}/biomass/predict",
            json={"species": species, "length_cm": length_cm, "girth_cm": girth_cm},
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return BiomassEstimationResult(
            species=payload["species"],
            biomass_kg=payload["biomass_kg"],
            method=payload["method"],
        )

    def predict_combined(self, image: Image.Image, length_cm: float, girth_cm: float | None, species: str | None = None) -> CombinedPredictionResult:
        response = requests.post(
            f"{self.base_url.rstrip('/')}/predict",
            json={
                "image_base64": self._image_payload(image),
                "species": species,
                "length_cm": length_cm,
                "girth_cm": girth_cm,
            },
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return CombinedPredictionResult(
            species=payload["species"],
            species_confidence=payload["species_confidence"],
            biomass_kg=payload["biomass_kg"],
            biomass_method=payload["biomass_method"],
            species_scores=payload["species_scores"],
            is_reliable=payload.get("is_reliable", True),
            warning=payload.get("warning"),
            top_alternatives=payload.get("top_alternatives", []),
        )

