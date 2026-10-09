from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .biomass import estimate_biomass
from .config import DEFAULT_BIOMASS_MODEL_PATH, DEFAULT_SPECIES_MODEL_PATH
from .model import BiomassModelBundle, SpeciesModelBundle, load_bundle, validate_image
from .species_intelligence import SpeciesProfile, get_species_profile


@dataclass
class SpeciesIdentificationResult:
    species: str
    confidence: float
    scores: dict[str, float]
    is_reliable: bool = True
    uncertainty_status: str = "CONFIDENT"  # "CONFIDENT", "MARGINAL", "UNCERTAIN"
    margin_to_second: float = 0.0
    entropy: float = 0.0
    closed_world_caveat: str = (
        "Predictions are strictly restricted to the 5 supported classes (carp, catfish, milkfish, pangasius, tilapia). "
        "Confidence is not proof that the image contains a supported species; unsupported subjects will be forced into these categories."
    )
    top_alternatives: list[tuple[str, float]] = field(default_factory=list)
    warning: str | None = None
    species_profile: SpeciesProfile | None = None
    calibration_method: str = "Temperature Scaling (T = 2.2819, out-of-fold calibrated)"


@dataclass
class BiomassEstimationResult:
    species: str
    biomass_kg: float
    method: str
    is_experimental: bool = False
    citation: str = ""
    notes: str = ""
    length_convention: str = "Total Length (TL, cm)"
    population_context: str = ""
    limitations: str = ""


class SpeciesIdentificationService:
    def __init__(self, species_model_path: Path = DEFAULT_SPECIES_MODEL_PATH) -> None:
        species_bundle = load_bundle(species_model_path)
        self.species_model = (
            species_bundle
            if isinstance(species_bundle, SpeciesModelBundle)
            else SpeciesModelBundle(None, [], 0)
        )
        # Activate verified temperature scaling calibration (T = 2.2819) if supported
        if self.species_model.pipeline is not None and hasattr(self.species_model.pipeline, "temperature"):
            self.species_model.pipeline.temperature = 2.2819

    def is_ready(self) -> bool:
        return self.species_model.pipeline is not None

    @property
    def is_available(self) -> bool:
        return self.is_ready()

    @property
    def classes(self) -> list[str]:
        return [str(c) for c in self.species_model.class_names]

    def predict(
        self,
        image: Image.Image | str | Path,
        confidence_threshold: float = 0.35,
    ) -> SpeciesIdentificationResult:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        if self.species_model.pipeline is None:
            return SpeciesIdentificationResult(
                species="unknown",
                confidence=0.0,
                scores={"unknown": 1.0},
                is_reliable=False,
                uncertainty_status="UNCERTAIN",
                margin_to_second=0.0,
                entropy=0.0,
                top_alternatives=[],
                warning="Species model is offline. No trained weights found.",
                species_profile=None,
            )

        validate_image(image)
        species, confidence, scores = self.species_model.predict(image)

        # Sort all alternatives
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_alternatives = [(k, v) for k, v in sorted_scores if k != species][:3]

        # Calculate margin to second class and entropy
        second_prob = top_alternatives[0][1] if top_alternatives else 0.0
        margin_to_second = max(0.0, confidence - second_prob)

        import math
        entropy = -sum(
            p * math.log2(max(p, 1e-9)) for p in scores.values() if p > 0.0
        )

        # Check for uninformative, degenerate, or blank images (near-zero spatial contrast)
        img_arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        spatial_std = float(np.mean([img_arr[:, :, c].std() for c in range(3)]))
        is_flat_or_blank = spatial_std < 5.0

        # Multi-metric uncertainty classification with cautious, honest messaging
        if is_flat_or_blank:
            uncertainty_status = "UNCERTAIN"
            is_reliable = False
            warning = (
                f"Image lacks sufficient spatial texture or edge contrast (mean spatial std: {spatial_std:.2f}). "
                "Potential non-fish, blank, or uniform pattern. The model cannot identify morphological features."
            )
        elif confidence < 0.40 or margin_to_second < 0.10 or entropy > 1.95:
            uncertainty_status = "UNCERTAIN"
            is_reliable = False
            warning = (
                f"Prediction is UNCERTAIN (model confidence {confidence:.1%}, runner-up margin {margin_to_second:.1%}, entropy {entropy:.2f} bits). "
                "Morphological features are ambiguous or outside typical distributions. Manual verification is required before taking operational action."
            )
        elif confidence < 0.65 or margin_to_second < 0.25:
            uncertainty_status = "MARGINAL"
            is_reliable = False
            alt_name = top_alternatives[0][0] if top_alternatives else "next class"
            alt_prob = top_alternatives[0][1] if top_alternatives else 0.0
            warning = (
                f"Prediction has MODERATE confidence ({confidence:.1%}, margin to runner-up {margin_to_second:.1%}). "
                f"Significant probability mass on {alt_name.capitalize()} ({alt_prob:.1%}). Confirm anatomical traits before taking action."
            )
        else:
            uncertainty_status = "CONFIDENT"
            is_reliable = True
            warning = None

        profile = get_species_profile(species)

        return SpeciesIdentificationResult(
            species=species,
            confidence=confidence,
            scores=scores,
            is_reliable=is_reliable,
            uncertainty_status=uncertainty_status,
            margin_to_second=round(margin_to_second, 4),
            entropy=round(entropy, 3),
            top_alternatives=top_alternatives,
            warning=warning,
            species_profile=profile,
        )

    def identify(
        self,
        image: Image.Image | str | Path,
        confidence_threshold: float = 0.28,
    ) -> SpeciesIdentificationResult:
        return self.predict(image, confidence_threshold=confidence_threshold)


class BiomassEstimationService:
    def __init__(self, biomass_model_path: Path = DEFAULT_BIOMASS_MODEL_PATH) -> None:
        biomass_bundle = load_bundle(biomass_model_path)
        self.biomass_model = (
            biomass_bundle
            if isinstance(biomass_bundle, BiomassModelBundle)
            else BiomassModelBundle(None)
        )

    def is_ready(self) -> bool:
        return self.biomass_model.pipeline is not None

    def predict(
        self,
        species: str,
        length_cm: float,
        girth_cm: float | None = None,
        use_ml_regressor: bool = False,
    ) -> BiomassEstimationResult:
        if not species or not species.strip():
            raise ValueError("Species is required for biomass estimation")
        if length_cm <= 0:
            raise ValueError("Length must be greater than zero")
        if girth_cm is not None and girth_cm < 0:
            raise ValueError("Girth cannot be negative")

        # Ground truth: Standard fisheries peer-reviewed power equations (Froese 2006, FishBase)
        formula_est = estimate_biomass(species, length_cm, girth_cm)

        if use_ml_regressor and self.biomass_model.pipeline is not None:
            biomass_value = self.biomass_model.predict(species, length_cm, girth_cm)
            return BiomassEstimationResult(
                species=species,
                biomass_kg=round(biomass_value, 4),
                method="experimental_trained_regressor",
                is_experimental=True,
                citation="Experimental gradient boosting regressor fitted on 20 sample records.",
                notes="Trained on a 20-sample demo dataset; experimental only.",
            )

        return BiomassEstimationResult(
            species=species,
            biomass_kg=formula_est.estimated_biomass_kg,
            method=formula_est.method,
            is_experimental=formula_est.is_experimental,
            citation=formula_est.citation,
            notes=formula_est.notes,
            length_convention=formula_est.length_convention,
            population_context=formula_est.population_context,
            limitations=formula_est.limitations,
        )


@dataclass
class CombinedPredictionResult:
    species: str
    species_confidence: float
    biomass_kg: float
    biomass_method: str
    species_scores: dict[str, float]
    is_reliable: bool = True
    top_alternatives: list[tuple[str, float]] = field(default_factory=list)
    warning: str | None = None
    species_profile: SpeciesProfile | None = None


class FishPredictionService:
    def __init__(
        self,
        species_model_path: Path = DEFAULT_SPECIES_MODEL_PATH,
        biomass_model_path: Path = DEFAULT_BIOMASS_MODEL_PATH,
    ) -> None:
        self.species_service = SpeciesIdentificationService(species_model_path)
        self.biomass_service = BiomassEstimationService(biomass_model_path)

    def predict(
        self,
        image: Image.Image,
        length_cm: float,
        girth_cm: float | None,
        species_override: str | None = None,
    ) -> CombinedPredictionResult:
        species_result = self.species_service.predict(image)
        chosen_species = species_override or species_result.species
        biomass_result = self.biomass_service.predict(chosen_species, length_cm, girth_cm)

        return CombinedPredictionResult(
            species=chosen_species,
            species_confidence=species_result.confidence,
            biomass_kg=biomass_result.biomass_kg,
            biomass_method=biomass_result.method,
            species_scores=species_result.scores,
            is_reliable=species_result.is_reliable,
            top_alternatives=species_result.top_alternatives,
            warning=species_result.warning,
            species_profile=species_result.species_profile,
        )