from __future__ import annotations

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from fishfarm_ai.config import DEFAULT_FISHBASE_PATH
from fishfarm_ai.fishbase_service import FishBaseRecord, FishBaseService, get_fishbase_service


class TestFishBaseService(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = FishBaseService(DEFAULT_FISHBASE_PATH)

    def test_service_availability_and_total_count(self) -> None:
        self.assertTrue(self.service.is_available, "FishBase dataset should be loaded and available")
        self.assertEqual(self.service.total_count, 36132, "Total species count should equal 36,132")

    def test_scientific_name_search(self) -> None:
        # Case-insensitive scientific binomial search
        results = self.service.search("chanos chanos", limit=10)
        self.assertGreater(len(results), 0)
        rec = results[0]
        self.assertEqual(rec.scientific_name, "Chanos chanos")
        self.assertEqual(rec.genus, "Chanos")
        self.assertEqual(rec.species, "chanos")
        self.assertEqual(rec.common_name, "Milkfish")
        self.assertIn("Marine / Saltwater", rec.habitats)

    def test_genus_search(self) -> None:
        # Search by genus
        results = self.service.search("Oreochromis", limit=20)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].genus.lower(), "oreochromis")
        for r in results:
            self.assertIn("oreochromis", r.genus.lower())

    def test_common_name_search(self) -> None:
        # Case-insensitive common name search
        results = self.service.search("Giant salmon carp", limit=10)
        self.assertGreater(len(results), 0)
        names = [r.scientific_name for r in results]
        self.assertIn("Aaptosyax grypus", names)

    def test_missing_common_name_handling(self) -> None:
        # Aapticheilichthys websteri has SpecCode 64588 and NaN for FBname
        rec = self.service.get_by_spec_code(64588)
        self.assertIsNotNone(rec)
        self.assertIsNone(rec.common_name)
        # display_title falls back to scientific name
        self.assertEqual(rec.display_title, "Aapticheilichthys websteri")

    def test_no_results_handling(self) -> None:
        # Query that matches nothing returns empty list gracefully
        results = self.service.search("XyZzQ_NonExistentFish_9999", limit=10)
        self.assertEqual(results, [])

    def test_result_limits(self) -> None:
        # Verify limit capping
        res_5 = self.service.search("carp", limit=5)
        self.assertEqual(len(res_5), 5)
        res_12 = self.service.search("carp", limit=12)
        self.assertEqual(len(res_12), 12)

    def test_aquaculture_filter(self) -> None:
        # Verify aquaculture filtering
        commercial_results = self.service.search(query="", limit=50, aquaculture_only=True)
        self.assertGreater(len(commercial_results), 0)
        for r in commercial_results:
            self.assertIn(r.aquaculture_use, ["commercial", "experimental", "likely future use"])

    def test_family_code_representation(self) -> None:
        # Verify family code is integer FamCode, not claimed to be a name string
        rec = self.service.get_by_spec_code(1450)  # Cyprinus carpio
        self.assertIsNotNone(rec)
        self.assertIsInstance(rec.family_code, int)
        self.assertEqual(rec.family_code, 122)

    def test_missing_dataset_file_handling(self) -> None:
        # Graceful handling when parquet path does not exist
        missing_service = FishBaseService("data/fishbase/non_existent_file.parquet")
        self.assertFalse(missing_service.is_available)
        self.assertEqual(missing_service.total_count, 0)
        self.assertIsNotNone(missing_service.load_error)
        results = missing_service.search("tilapia")
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
