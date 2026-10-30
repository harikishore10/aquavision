"""Gemini Vision Second Opinion Service for AquaVision.

Provides an optional, non-intrusive cloud AI consultation for uploaded fish images
using the Google Gemini Vision API via standard HTTPS REST requests.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import io
import json
import os
from typing import Any

from PIL import Image
import requests


SUPPORTED_FARM_SPECIES = ["tilapia", "catfish", "carp", "milkfish", "pangasius"]

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


@dataclass
class GeminiOpinionResult:
    status: str  # "SUCCESS", "ERROR", "DISABLED", "BILLING_BLOCKED", "QUOTA_EXCEEDED"
    species_name: str = ""
    scientific_name: str = ""
    confidence: str = ""  # "High", "Moderate", "Low"
    is_supported_farm_species: bool = False
    matched_supported_species: str | None = None
    anatomical_observations: list[str] = field(default_factory=list)
    explanation: str = ""
    notes: str = ""
    error_message: str | None = None
    raw_response: str = ""

    @property
    def is_success(self) -> bool:
        return self.status == "SUCCESS"


class GeminiVisionService:
    """Service to obtain an optional second opinion on aquatic organism images from Gemini Vision."""

    def __init__(self, default_model: str = DEFAULT_GEMINI_MODEL, timeout_seconds: float = 20.0) -> None:
        self.default_model = default_model
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def get_api_key(explicit_key: str | None = None) -> str | None:
        """Retrieve API key from explicit parameter, environment, or Streamlit secrets."""
        if explicit_key and explicit_key.strip():
            return explicit_key.strip()

        env_key = os.getenv("GEMINI_API_KEY", "").strip()
        if env_key:
            return env_key

        # Try Streamlit secrets if running inside Streamlit
        try:
            import streamlit as st

            if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                return str(st.secrets["GEMINI_API_KEY"]).strip()
        except Exception:
            pass

        return None

    @classmethod
    def get_model_name(cls, explicit_model: str | None = None) -> str:
        """Retrieve model name from explicit parameter, environment, or Streamlit secrets."""
        if explicit_model and explicit_model.strip():
            return explicit_model.strip()

        env_model = os.getenv("GEMINI_MODEL", "").strip()
        if env_model:
            return env_model

        try:
            import streamlit as st

            if hasattr(st, "secrets") and "GEMINI_MODEL" in st.secrets:
                return str(st.secrets["GEMINI_MODEL"]).strip()
        except Exception:
            pass

        return DEFAULT_GEMINI_MODEL

    def is_available(self, explicit_key: str | None = None) -> bool:
        """Check if an API key is configured."""
        return self.get_api_key(explicit_key) is not None

    def _prepare_image_payload(self, image: Image.Image) -> tuple[str, str]:
        """Convert PIL image to base64 JPEG bytes."""
        rgb_img = image.convert("RGB")
        # Resize to max 1024 on longest edge for efficient transmission
        max_dim = 1024
        if max(rgb_img.width, rgb_img.height) > max_dim:
            rgb_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        rgb_img.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded, "image/jpeg"

    def analyze_image(
        self,
        image: Image.Image,
        api_key: str | None = None,
        model: str | None = None,
    ) -> GeminiOpinionResult:
        """Query Gemini Vision for a biological second opinion on the image."""
        resolved_key = self.get_api_key(api_key)
        if not resolved_key:
            return GeminiOpinionResult(
                status="DISABLED",
                error_message="Assessment API key is not configured. Set the GEMINI_API_KEY environment variable or enter it in the sidebar.",
            )

        selected_model = self.get_model_name(model or (self.default_model if self.default_model != DEFAULT_GEMINI_MODEL else None))
        url = f"{GEMINI_API_URL_TEMPLATE.format(model=selected_model)}?key={resolved_key}"

        try:
            b64_data, mime_type = self._prepare_image_payload(image)
        except Exception as exc:
            return GeminiOpinionResult(
                status="ERROR",
                error_message=f"Failed to process image for Gemini API: {exc}",
            )

        prompt_text = (
            "You are an expert aquaculture biologist and ichthyologist acting as an independent advisory consultant.\n"
            "Examine this image of an aquatic organism.\n\n"
            "Our primary local machine learning model operates under a closed-world assumption restricted to 5 supported classes:\n"
            "1. Nile Tilapia (Oreochromis niloticus)\n"
            "2. African Sharptooth Catfish (Clarias gariepinus)\n"
            "3. Common Carp (Cyprinus carpio)\n"
            "4. Milkfish (Chanos chanos)\n"
            "5. Striped Catfish / Pangasius (Pangasianodon hypophthalmus)\n\n"
            "Provide your biological assessment as valid JSON with these exact keys:\n"
            "{\n"
            '  "species_name": "Common name of organism",\n'
            '  "scientific_name": "Latin scientific binomial",\n'
            '  "confidence": "High" | "Moderate" | "Low",\n'
            '  "is_supported_farm_species": true | false,\n'
            '  "matched_supported_species": "tilapia" | "catfish" | "carp" | "milkfish" | "pangasius" | null,\n'
            '  "anatomical_observations": ["Point 1 about fin/body/barbels/scales", "Point 2"],\n'
            '  "explanation": "Brief biological justification of your identification",\n'
            '  "notes": "Any note regarding image clarity, non-fish status, or open-world divergence"\n'
            "}\n\n"
            "Do not output markdown code blocks. Return only raw JSON."
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_data,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 800,
            },
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout_seconds,
            )
        except requests.exceptions.Timeout:
            return GeminiOpinionResult(
                status="ERROR",
                error_message="Assessment request timed out (server did not respond within timeout limit).",
            )
        except requests.exceptions.ConnectionError:
            return GeminiOpinionResult(
                status="ERROR",
                error_message="Network error: Unable to connect to remote assessment service. Please check your internet connection.",
            )
        except Exception as exc:
            return GeminiOpinionResult(
                status="ERROR",
                error_message=f"Request error: {exc}",
            )

        # Handle HTTP status codes gracefully
        if response.status_code == 404:
            return GeminiOpinionResult(
                status="ERROR",
                error_message=(
                    f"Model '{selected_model}' not found (HTTP 404). "
                    "Please ensure this model ID is enabled and available for your project and API key."
                ),
            )

        if response.status_code == 429:
            return GeminiOpinionResult(
                status="QUOTA_EXCEEDED",
                error_message="Assessment service quota or rate limit exceeded (Resource Exhausted). Please wait before trying again.",
            )

        if response.status_code in (401, 403):
            err_text = response.text.lower()
            if "billing" in err_text or "billing_not_found" in err_text:
                return GeminiOpinionResult(
                    status="BILLING_BLOCKED",
                    error_message=(
                        "Remote assessment service blocked access due to billing requirements for this project or model. "
                        "Paid billing has NOT been enabled. The local model will continue operating normally."
                    ),
                )
            return GeminiOpinionResult(
                status="ERROR",
                error_message="Assessment service access denied. Please verify your API key permissions.",
            )

        if response.status_code != 200:
            return GeminiOpinionResult(
                status="ERROR",
                error_message=f"Assessment service returned HTTP {response.status_code}: {response.text[:300]}",
            )

        # Parse Gemini response payload
        try:
            resp_json = response.json()
            candidates = resp_json.get("candidates", [])
            if not candidates:
                return GeminiOpinionResult(
                    status="ERROR",
                    error_message="Gemini API returned an empty response (no candidate text).",
                )

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return GeminiOpinionResult(
                    status="ERROR",
                    error_message="Gemini API returned candidate without content parts.",
                )

            raw_text = parts[0].get("text", "").strip()
            return self._parse_gemini_json(raw_text)

        except Exception as exc:
            return GeminiOpinionResult(
                status="ERROR",
                error_message=f"Failed to parse Gemini response: {exc}",
            )

    def _parse_gemini_json(self, raw_text: str) -> GeminiOpinionResult:
        """Parse JSON response from Gemini, removing any markdown code formatting if present."""
        cleaned_text = raw_text.strip()
        if cleaned_text.startswith("```"):
            lines = cleaned_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned_text = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned_text)
            species_name = str(data.get("species_name", "Unknown Organism"))
            scientific_name = str(data.get("scientific_name", ""))
            confidence = str(data.get("confidence", "Moderate"))
            is_supported = bool(data.get("is_supported_farm_species", False))
            matched_species = data.get("matched_supported_species")
            if matched_species:
                matched_species = str(matched_species).lower().strip()
                if matched_species not in SUPPORTED_FARM_SPECIES:
                    matched_species = None

            observations = data.get("anatomical_observations", [])
            if not isinstance(observations, list):
                observations = [str(observations)]
            observations = [str(o) for o in observations]

            explanation = str(data.get("explanation", ""))
            notes = str(data.get("notes", ""))

            return GeminiOpinionResult(
                status="SUCCESS",
                species_name=species_name,
                scientific_name=scientific_name,
                confidence=confidence,
                is_supported_farm_species=is_supported,
                matched_supported_species=matched_species,
                anatomical_observations=observations,
                explanation=explanation,
                notes=notes,
                raw_response=raw_text,
            )
        except json.JSONDecodeError:
            # Fallback if raw text wasn't strict JSON
            return GeminiOpinionResult(
                status="SUCCESS",
                species_name="Identified Aquatic Organism",
                explanation=raw_text[:500],
                notes="Response received in freeform text format.",
                raw_response=raw_text,
            )
