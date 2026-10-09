from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_SRC = Path(__file__).resolve().parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from fishfarm_ai.services import BiomassEstimationService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fish biomass estimation")
    parser.add_argument("--species", type=str, required=True, help="Fish species label")
    parser.add_argument("--length-cm", type=float, required=True, help="Measured fish length in centimeters")
    parser.add_argument("--girth-cm", type=float, default=None, help="Measured fish girth in centimeters")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = BiomassEstimationService()
    result = service.predict(args.species, args.length_cm, args.girth_cm)
    status_label = "EXPERIMENTAL" if getattr(result, "is_experimental", False) else "VERIFIED"
    print(f"Species:              {result.species.capitalize()}")
    print(f"Estimated Biomass:    {result.biomass_kg:.3f} kg ({result.biomass_kg * 1000:.1f} g)")
    print(f"Estimation Method:    {result.method} [{status_label}]")
    if getattr(result, "citation", None):
        print(f"Scientific Citation:  {result.citation}")
    if getattr(result, "length_convention", None):
        print(f"Length Convention:    {result.length_convention}")
    if getattr(result, "population_context", None):
        print(f"Population Context:   {result.population_context}")
    if getattr(result, "limitations", None):
        print(f"Biological Limits:    {result.limitations}")
    if getattr(result, "notes", None):
        print(f"Method Notes:         {result.notes}")
    print("Physical Input Scope: Requires manual physical caliper/board metric measurements (cm); not image-based.")


if __name__ == "__main__":
    main()
