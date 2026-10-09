from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from fishfarm_ai.gemini_service import GeminiOpinionResult, GeminiVisionService


class TestGeminiVisionService(unittest.TestCase):
    def setUp(self) -> None:
        self.service = GeminiVisionService()
        self.sample_img = Image.new("RGB", (64, 64), color=(50, 100, 150))

    def test_disabled_without_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            res = self.service.analyze_image(self.sample_img, api_key=None)
            self.assertEqual(res.status, "DISABLED")
            self.assertFalse(res.is_success)
            self.assertIn("not configured", res.error_message or "")

    @patch("requests.post")
    def test_successful_gemini_response(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps({
                                    "species_name": "Nile Tilapia",
                                    "scientific_name": "Oreochromis niloticus",
                                    "confidence": "High",
                                    "is_supported_farm_species": True,
                                    "matched_supported_species": "tilapia",
                                    "anatomical_observations": [
                                        "Distinct vertical stripes on caudal fin",
                                        "Deep laterally compressed body",
                                    ],
                                    "explanation": "Clear diagnostic barring visible across body flank.",
                                    "notes": "Healthy specimen in clear water.",
                                })
                            }
                        ]
                    }
                }
            ]
        }
        mock_resp.json.return_value = mock_payload
        mock_post.return_value = mock_resp

        res = self.service.analyze_image(self.sample_img, api_key="test-api-key")
        self.assertEqual(res.status, "SUCCESS")
        self.assertTrue(res.is_success)
        self.assertEqual(res.species_name, "Nile Tilapia")
        self.assertEqual(res.scientific_name, "Oreochromis niloticus")
        self.assertEqual(res.confidence, "High")
        self.assertTrue(res.is_supported_farm_species)
        self.assertEqual(res.matched_supported_species, "tilapia")
        self.assertEqual(len(res.anatomical_observations), 2)
        self.assertIn("vertical stripes", res.anatomical_observations[0])

    @patch("requests.post")
    def test_markdown_code_fenced_json_parsing(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        fenced_json = (
            "```json\n"
            "{\n"
            '  "species_name": "Common Carp",\n'
            '  "scientific_name": "Cyprinus carpio",\n'
            '  "confidence": "Moderate",\n'
            '  "is_supported_farm_species": true,\n'
            '  "matched_supported_species": "carp",\n'
            '  "anatomical_observations": ["Pair of maxillary barbels"],\n'
            '  "explanation": "Barbels clearly visible near corners of upper mouth.",\n'
            '  "notes": ""\n'
            "}\n"
            "```"
        )
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": fenced_json}]}}]
        }
        mock_post.return_value = mock_resp

        res = self.service.analyze_image(self.sample_img, api_key="test-api-key")
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.species_name, "Common Carp")
        self.assertEqual(res.matched_supported_species, "carp")

    @patch("requests.post")
    def test_billing_blocked_handling(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Google API Error: Billing account not configured. Visit console to enable billing."
        mock_post.return_value = mock_resp

        res = self.service.analyze_image(self.sample_img, api_key="test-api-key")
        self.assertEqual(res.status, "BILLING_BLOCKED")
        self.assertFalse(res.is_success)
        self.assertIn("billing requirements", res.error_message or "")
        self.assertIn("Paid billing has NOT been enabled", res.error_message or "")

    @patch("requests.post")
    def test_quota_exceeded_handling(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Resource has been exhausted (e.g. check quota)."
        mock_post.return_value = mock_resp

        res = self.service.analyze_image(self.sample_img, api_key="test-api-key")
        self.assertEqual(res.status, "QUOTA_EXCEEDED")
        self.assertFalse(res.is_success)
        self.assertIn("quota or rate limit exceeded", res.error_message or "")

    @patch("requests.post")
    def test_default_model_and_endpoint(self, mock_post: MagicMock) -> None:
        from fishfarm_ai.gemini_service import DEFAULT_GEMINI_MODEL, GEMINI_API_URL_TEMPLATE

        self.assertEqual(DEFAULT_GEMINI_MODEL, "gemini-3.5-flash-lite")
        self.assertEqual(self.service.default_model, "gemini-3.5-flash-lite")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"species_name": "Tilapia"}'}]}}]
        }
        mock_post.return_value = mock_resp

        self.service.analyze_image(self.sample_img, api_key="secret-test-key")

        mock_post.assert_called_once()
        called_url = mock_post.call_args[0][0]
        expected_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key=secret-test-key"
        self.assertEqual(called_url, expected_url)

    @patch("requests.post")
    def test_model_not_found_404_handling(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = '{"error": {"code": 404, "message": "models/gemini-3.5-flash-lite is not found for API version v1beta"}}'
        mock_post.return_value = mock_resp

        res = self.service.analyze_image(self.sample_img, api_key="test-api-key")
        self.assertEqual(res.status, "ERROR")
        self.assertFalse(res.is_success)
        self.assertIn("HTTP 404", res.error_message or "")
        self.assertIn("gemini-3.5-flash-lite", res.error_message or "")


if __name__ == "__main__":
    unittest.main()

