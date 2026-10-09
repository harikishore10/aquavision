from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_SPECIES_MODEL_PATH = MODELS_DIR / "species_model.joblib"
DEFAULT_BIOMASS_MODEL_PATH = MODELS_DIR / "biomass_model.joblib"
DEFAULT_METADATA_PATH = MODELS_DIR / "metadata.json"

DEFAULT_SPECIES = ["tilapia", "catfish", "carp", "milkfish", "pangasius"]
DEFAULT_FISHBASE_PATH = PROJECT_ROOT / "data" / "fishbase" / "species.parquet"
