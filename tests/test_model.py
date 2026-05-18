import unittest
import os
import sys
import pickle
import mlflow
import mlflow.pyfunc
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants import (
    MODEL_EVALUATION_MODEL_NAME,
    MODEL_PUSHER_ALIAS,
)


class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """MLflow se champion model load karo"""

        # ── Auth ────────────────────────────────────────────────
        dagshub_token = os.getenv("DEEPSHIELD_TEST")
        if not dagshub_token:
            raise EnvironmentError("DEEPSHIELD_TEST env variable not set")

        # ── Constants validation ─────────────────────────────────
        # BUG FIX #3: constants empty hone par crash hoga silently — ab explicit error
        if not MODEL_EVALUATION_MODEL_NAME:
            raise EnvironmentError("MODEL_EVALUATION_MODEL_NAME is empty in constants")
        if not MODEL_PUSHER_ALIAS:
            raise EnvironmentError("MODEL_PUSHER_ALIAS is empty in constants")

        os.environ["MLFLOW_TRACKING_USERNAME"] = "kaushik-chariya"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_tracking_uri(
            "https://dagshub.com/kaushik-chariya/Deep-Shield-Mail.mlflow"
        )

        # ── Champion model load ──────────────────────────────────
        model_uri = f"models:/{MODEL_EVALUATION_MODEL_NAME}@{MODEL_PUSHER_ALIAS}"
        cls.model = mlflow.pyfunc.load_model(model_uri)

        # ── Preprocessor load ────────────────────────────────────
        # BUG FIX #1: mlflow.artifacts.load_artifact() exist nahi karta
        # Correct API: download_artifacts() → local path milta hai → pickle se load karo
        client = mlflow.MlflowClient()
        champion = client.get_model_version_by_alias(
            name=MODEL_EVALUATION_MODEL_NAME,
            alias=MODEL_PUSHER_ALIAS,
        )
        run_id = champion.run_id

        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/transformers/transformers.pkl"
        )
        with open(local_path, "rb") as f:
            cls.preprocessor = pickle.load(f)

        # ── Underlying sklearn model for probability tests ────────
        # BUG FIX #2: pyfunc model .predict() labels deta hai, probabilities nahi
        # sklearn model directly chahiye predict_proba ke liye
        model_download_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/model/model.pkl"
        )
        with open(model_download_path, "rb") as f:
            cls.sklearn_model = pickle.load(f)

    # ── Load tests ──────────────────────────────────────────────

    def test_model_loads(self):
        """Champion model MLflow se load hota hai"""
        self.assertIsNotNone(self.model)

    def test_preprocessor_loads(self):
        """Preprocessor MLflow se load hota hai"""
        self.assertIsNotNone(self.preprocessor)

    # ── Prediction tests ────────────────────────────────────────

    def test_model_predicts_spam(self):
        """Model spam email ko 1 predict karta hai"""
        sample = ["Congratulations! You have won a free lottery. Click here to claim now!"]
        transformed = self.preprocessor.transform(sample)
        prediction = self.model.predict(transformed)
        self.assertEqual(int(prediction[0]), 1,
            "Spam email ko 1 predict hona chahiye")

    def test_model_predicts_ham(self):
        """Model normal email ko 0 predict karta hai"""
        sample = ["Hi team, please review the attached report before tomorrow's meeting."]
        transformed = self.preprocessor.transform(sample)
        prediction = self.model.predict(transformed)
        self.assertEqual(int(prediction[0]), 0,
            "Normal email ko 0 predict hona chahiye")

    # ── Probability tests ───────────────────────────────────────
    # BUG FIX #2: pyfunc predict() se probability nahi milti
    # sklearn_model.predict_proba() use karo

    def test_spam_probability_high(self):
        """Spam email ki probability 0.7 se zyada honi chahiye"""
        sample = ["Win a free iPhone now! Limited time offer! Claim your prize!"]
        transformed = self.preprocessor.transform(sample)
        proba = self.sklearn_model.predict_proba(transformed)
        spam_prob = float(proba[0][1])  # index 1 = spam class probability
        self.assertGreater(spam_prob, 0.7,
            f"Spam probability 70% se zyada honi chahiye, got {spam_prob:.2f}")

    def test_ham_probability_high(self):
        """Normal email ki ham probability 0.7 se zyada honi chahiye"""
        sample = ["Hi team, please review the attached report before tomorrow's meeting."]
        transformed = self.preprocessor.transform(sample)
        proba = self.sklearn_model.predict_proba(transformed)
        spam_prob = float(proba[0][1])  # index 1 = spam class probability
        self.assertLess(spam_prob, 0.3,
            f"Ham email ki spam probability 30% se kam honi chahiye, got {spam_prob:.2f}")

    # ── Batch prediction test ───────────────────────────────────

    def test_model_batch_prediction(self):
        """Model multiple emails ek saath predict kar sakta hai"""
        samples = [
            "Congratulations! You have won a free lottery!",
            "Hi team, please review the report.",
            "Click here to claim your free prize now!",
            "Meeting is scheduled for tomorrow at 10am.",
        ]
        transformed = self.preprocessor.transform(samples)
        predictions = self.model.predict(transformed)
        self.assertEqual(len(predictions), 4)
        for pred in predictions:
            self.assertIn(int(pred), [0, 1])


if __name__ == "__main__":
    unittest.main()