from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from fishfarm_ai.config import DEFAULT_BIOMASS_MODEL_PATH, DEFAULT_SPECIES_MODEL_PATH
from fishfarm_ai.model import save_bundle, train_biomass_model, train_species_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fish species and biomass models")
    parser.add_argument("--species-data", type=Path, default=Path("data/species_images"), help="Root folder with images grouped by species")
    parser.add_argument("--biomass-data", type=Path, default=Path("data/sample_biomass_records.csv"), help="CSV with species, length_cm, girth_cm, biomass_kg")
    parser.add_argument("--output", type=Path, default=Path("models"), help="Output directory for trained bundles")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("AquaVision Model Training Pipeline")
    print(f"Species Data: {args.species_data}")
    print(f"Biomass Data: {args.biomass_data}")
    print(f"Output Dir:   {args.output}")
    print("==================================================")

    # 1. Train species classification model
    species_bundle = train_species_model(args.species_data)
    save_bundle(species_bundle, args.output / DEFAULT_SPECIES_MODEL_PATH.name)
    print(f"\n[OK] Saved species model bundle to {args.output / DEFAULT_SPECIES_MODEL_PATH.name}")

    # 2. Train biomass regression model
    biomass_records = pd.read_csv(args.biomass_data)
    biomass_bundle = train_biomass_model(biomass_records)
    save_bundle(biomass_bundle, args.output / DEFAULT_BIOMASS_MODEL_PATH.name)
    print(f"[OK] Saved biomass model bundle to {args.output / DEFAULT_BIOMASS_MODEL_PATH.name}")

    # 3. Export detailed evaluation report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_source": "GBIF Open Biodiversity Data (iNaturalist research-grade observations)",
        "license": "Creative Commons (CC-BY / CC0 / CC BY-NC)",
        "species_model": {
            "classes": species_bundle.class_names,
            "feature_count": species_bundle.feature_count,
            "metrics": species_bundle.evaluation_metrics,
        },
        "biomass_model": {
            "metrics": biomass_bundle.evaluation_metrics,
        },
    }
    report_path = args.output / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[OK] Exported evaluation report to {report_path}")
    print("\nTraining workflow completed successfully!")


if __name__ == "__main__":
    main()
