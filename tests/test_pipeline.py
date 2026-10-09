from __future__ import annotations

import json
import unittest
from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from fishfarm_ai.biomass import estimate_batch_biomass, estimate_biomass
from fishfarm_ai.model import load_bundle, validate_image
from fishfarm_ai.services import SpeciesIdentificationService
from fishfarm_ai.species_intelligence import get_species_profile, search_species_profiles


class TestAquaVisionPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.species_service = SpeciesIdentificationService()

    def test_model_loading(self) -> None:
        self.assertTrue(self.species_service.is_ready(), "Species model should be loaded and ready")
        expected_classes = {"carp", "catfish", "milkfish", "pangasius", "tilapia"}
        self.assertEqual(set(self.species_service.classes), expected_classes)

    def test_species_inference_on_training_fixtures(self) -> None:
        """Verify inference on images confirmed to belong to the training partition, preserving test set isolation."""
        eval_report_path = PROJECT_ROOT / "models" / "evaluation_report.json"
        self.assertTrue(eval_report_path.exists(), "Evaluation report must exist")

        with open(eval_report_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        test_files = set(eval_data.get("species_model", {}).get("metrics", {}).get("partition_manifest", {}).get("test_files", []))
        self.assertEqual(len(test_files), 125, "Held-out test set must contain exactly 125 files")

        # Confirm replacement images strictly belong to training partition (NOT in test_files)
        self.assertNotIn("carp_001.jpg", test_files, "carp_001.jpg must belong to the training partition")
        carp_train_path = PROJECT_ROOT / "data" / "species_images" / "carp" / "carp_001.jpg"
        carp_res = self.species_service.predict(carp_train_path)
        self.assertEqual(carp_res.species, "carp")
        self.assertGreater(carp_res.confidence, 0.40)
        self.assertIsNotNone(carp_res.species_profile)

        self.assertNotIn("tilapia_001.jpg", test_files, "tilapia_001.jpg must belong to the training partition")
        tilapia_train_path = PROJECT_ROOT / "data" / "species_images" / "tilapia" / "tilapia_001.jpg"
        tilapia_res = self.species_service.predict(tilapia_train_path)
        self.assertEqual(tilapia_res.species, "tilapia")
        self.assertGreater(tilapia_res.confidence, 0.50)

    def test_invalid_image_handling(self) -> None:
        with self.assertRaises((TypeError, AttributeError)):
            validate_image("not_an_image")  # type: ignore

        tiny_img = Image.new("RGB", (10, 10), color=(128, 128, 128))
        with self.assertRaises(ValueError):
            validate_image(tiny_img)

    def test_species_intelligence_retrieval(self) -> None:
        species_list = ["tilapia", "catfish", "carp", "milkfish", "pangasius"]
        for sp in species_list:
            profile = get_species_profile(sp)
            self.assertIsNotNone(profile, f"Species profile missing for {sp}")
            self.assertTrue(len(profile.scientific_name) > 0)
            self.assertTrue(len(profile.water_quality_parameters) >= 3)
            self.assertTrue(len(profile.references) >= 2)

    def test_biomass_and_batch_calculation(self) -> None:
        # Verified length-only FishBase formula: W = 0.0177 * 25^2.99 ≈ 0.264 kg
        length_only = estimate_biomass("tilapia", 25.0)
        self.assertEqual(length_only.method, "verified_fishbase_length_weight")
        self.assertFalse(length_only.is_experimental)
        self.assertAlmostEqual(length_only.estimated_biomass_kg, 0.264, delta=0.03)

        # Experimental length-girth formula: W = (25 * 17.5^2) / 29.0 ≈ 0.264 kg
        length_girth = estimate_biomass("tilapia", 25.0, 17.5)
        self.assertEqual(length_girth.method, "experimental_length_girth_morphology")
        self.assertTrue(length_girth.is_experimental)
        self.assertAlmostEqual(length_girth.estimated_biomass_kg, 0.264, delta=0.03)

        # Batch aggregation
        batch = estimate_batch_biomass(
            species="tilapia",
            individual_weight_kg=length_only.estimated_biomass_kg,
            fish_count=1000,
            survival_rate_percent=90.0,
        )
        self.assertEqual(batch.effective_fish_count, 900)
        self.assertAlmostEqual(batch.total_biomass_kg, 237.6, delta=10.0)
        self.assertGreater(batch.estimated_daily_feed_kg, 1.0)

    def test_uncertainty_handling(self) -> None:
        flat_img = Image.new("RGB", (128, 128), color=(128, 128, 128))
        result = self.species_service.predict(flat_img)

        self.assertIn(result.uncertainty_status, {"UNCERTAIN", "MARGINAL"})
        self.assertFalse(result.is_reliable)
        self.assertIsNotNone(result.warning)
        self.assertIn("strictly restricted to the 5 supported classes", result.closed_world_caveat)


if __name__ == "__main__":
    unittest.main()
