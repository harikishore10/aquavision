from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image

PROJECT_SRC = Path(__file__).resolve().parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from fishfarm_ai.services import SpeciesIdentificationService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fish species identification")
    parser.add_argument("image_pos", nargs="?", type=Path, default=None, help="Input fish image file path")
    parser.add_argument("--image", type=Path, default=None, help="Input fish image file path (optional flag)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image or args.image_pos
    if not image_path:
        print("ERROR: Please specify an image path: python predict.py <image_path>")
        sys.exit(1)
    if not image_path.exists():
        print(f"ERROR: Image file not found: {image_path}")
        sys.exit(1)
    image = Image.open(image_path)
    service = SpeciesIdentificationService()

    if not service.is_ready():
        print("ERROR: Species classifier is offline. No trained weights found at models/species_model.joblib.")
        sys.exit(1)

    result = service.predict(image)
    print(f"Predicted Species:  {result.species.capitalize()}")
    print(f"Model Confidence:   {result.confidence:.2%}")
    print(f"Calibration Method: {getattr(result, 'calibration_method', 'Uncalibrated')}")
    print(f"Uncertainty Status: {result.uncertainty_status} (Margin: {result.margin_to_second:.1%}, Entropy: {result.entropy:.2f} bits)")
    print(f"Reliability Status: {'High Confidence (Biological verification recommended)' if result.is_reliable else 'Low/Moderate Confidence (Manual Verification Required)'}")

    if result.warning:
        print(f"Advisory Warning:   {result.warning}")

    print(f"Closed-World Scope: {result.closed_world_caveat}")

    if result.top_alternatives:
        alts = ", ".join(f"{name.capitalize()} ({prob:.1%})" for name, prob in result.top_alternatives)
        print(f"Top Alternatives:   {alts}")

    if result.species_profile:
        print(f"Scientific Name:    {result.species_profile.scientific_name}")
        print(f"Target Harvest Wt:  {result.species_profile.target_harvest_weight}")


if __name__ == "__main__":
    main()
