from __future__ import annotations

import base64
from io import BytesIO

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from .auth import get_expected_api_token, get_expected_password, get_expected_username, require_api_token
from .services import BiomassEstimationService, FishPredictionService, SpeciesIdentificationService

app = FastAPI(title="Fish Aquaculture AI API", version="0.1.0")

species_service = SpeciesIdentificationService()
biomass_service = BiomassEstimationService()
combined_service = FishPredictionService()


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class SpeciesPredictRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image bytes")


class BiomassPredictRequest(BaseModel):
    species: str
    length_cm: float
    girth_cm: float | None = None


class CombinedPredictRequest(BaseModel):
    image_base64: str
    species: str | None = None
    length_cm: float
    girth_cm: float | None = None


class SpeciesPredictResponse(BaseModel):
    species: str
    confidence: float
    scores: dict[str, float]
    is_reliable: bool = True
    uncertainty_status: str = "CONFIDENT"
    margin_to_second: float = 0.0
    entropy: float = 0.0
    closed_world_caveat: str = ""
    warning: str | None = None
    top_alternatives: list[tuple[str, float]] = []
    calibration_method: str = "Temperature Scaling (T = 2.2819)"


class BiomassPredictResponse(BaseModel):
    species: str
    biomass_kg: float
    method: str
    is_experimental: bool = False
    citation: str = ""
    notes: str = ""


class CombinedPredictResponse(BaseModel):
    species: str
    species_confidence: float
    biomass_kg: float
    biomass_method: str
    species_scores: dict[str, float]
    is_reliable: bool = True
    uncertainty_status: str = "CONFIDENT"
    margin_to_second: float = 0.0
    entropy: float = 0.0
    closed_world_caveat: str = ""
    warning: str | None = None
    top_alternatives: list[tuple[str, float]] = []
    is_experimental_biomass: bool = False
    biomass_citation: str = ""
    calibration_method: str = "Temperature Scaling (T = 2.2819)"


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "auth": "enabled", "token_hint": "set FISHFARM_API_TOKEN"}


@app.post("/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    if payload.username != get_expected_username() or payload.password != get_expected_password():
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return LoginResponse(access_token=get_expected_api_token())


@app.post("/species/predict", response_model=SpeciesPredictResponse)
def predict_species(payload: SpeciesPredictRequest, _: str = Depends(require_api_token)) -> SpeciesPredictResponse:
    image = _decode_image(payload.image_base64)
    result = species_service.predict(image)
    return SpeciesPredictResponse(
        species=result.species,
        confidence=result.confidence,
        scores=result.scores,
        is_reliable=result.is_reliable,
        uncertainty_status=result.uncertainty_status,
        margin_to_second=result.margin_to_second,
        entropy=result.entropy,
        closed_world_caveat=result.closed_world_caveat,
        warning=result.warning,
        top_alternatives=result.top_alternatives,
        calibration_method=getattr(result, "calibration_method", "Temperature Scaling (T = 2.2819)"),
    )


@app.post("/biomass/predict", response_model=BiomassPredictResponse)
def predict_biomass(payload: BiomassPredictRequest, _: str = Depends(require_api_token)) -> BiomassPredictResponse:
    try:
        result = biomass_service.predict(payload.species, payload.length_cm, payload.girth_cm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BiomassPredictResponse(
        species=result.species,
        biomass_kg=result.biomass_kg,
        method=result.method,
        is_experimental=result.is_experimental,
        citation=result.citation,
        notes=result.notes,
    )


@app.post("/predict", response_model=CombinedPredictResponse)
def predict_combined(payload: CombinedPredictRequest, _: str = Depends(require_api_token)) -> CombinedPredictResponse:
    image = _decode_image(payload.image_base64)
    image_result = species_service.predict(image)
    biomass_input_species = payload.species or image_result.species
    try:
        biomass_result = biomass_service.predict(biomass_input_species, payload.length_cm, payload.girth_cm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CombinedPredictResponse(
        species=image_result.species,
        species_confidence=image_result.confidence,
        biomass_kg=biomass_result.biomass_kg,
        biomass_method=biomass_result.method,
        species_scores=image_result.scores,
        is_reliable=image_result.is_reliable,
        uncertainty_status=image_result.uncertainty_status,
        margin_to_second=image_result.margin_to_second,
        entropy=image_result.entropy,
        closed_world_caveat=image_result.closed_world_caveat,
        warning=image_result.warning,
        top_alternatives=image_result.top_alternatives,
        is_experimental_biomass=biomass_result.is_experimental,
        biomass_citation=biomass_result.citation,
        calibration_method=getattr(image_result, "calibration_method", "Temperature Scaling (T = 2.2819)"),
    )



def _decode_image(image_base64: str) -> Image.Image:
    try:
        raw_bytes = base64.b64decode(image_base64)
        return Image.open(BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image payload") from exc
