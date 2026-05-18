import unittest
import json
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.api.app import app


class TestFlaskApp(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    # ── /health ───────────────────────────────────────────────
    def test_health_returns_200(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_health_response_body(self):
        response = self.client.get("/health")
        data = json.loads(response.data)
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "Deep-Shield-Mail")

    # ── / (index) ─────────────────────────────────────────────
    def test_index_returns_200(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    # ── /predict ──────────────────────────────────────────────
    @patch("serving.api.app.get_pipeline")
    def test_predict_spam(self, mock_get_pipeline):
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = {
            "label": "Spam",
            "prediction": 1,
            "probability": 0.97
        }
        mock_get_pipeline.return_value = mock_pipeline

        response = self.client.post(
            "/predict",
            data=json.dumps({"email": "Congratulations! You won a free lottery!"}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("label", data)
        self.assertIn("prediction", data)
        self.assertIn("probability", data)
        self.assertEqual(data["label"], "Spam")
        self.assertEqual(data["prediction"], 1)

    @patch("serving.api.app.get_pipeline")
    def test_predict_not_spam(self, mock_get_pipeline):
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = {
            "label": "Not Spam",
            "prediction": 0,
            "probability": 0.02
        }
        mock_get_pipeline.return_value = mock_pipeline

        response = self.client.post(
            "/predict",
            data=json.dumps({"email": "Hey, are we still meeting tomorrow at 3pm?"}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["label"], "Not Spam")
        self.assertEqual(data["prediction"], 0)

    def test_predict_empty_email(self):
        response = self.client.post(
            "/predict",
            data=json.dumps({"email": ""}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)

    def test_predict_missing_email_field(self):
        response = self.client.post(
            "/predict",
            data=json.dumps({}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()