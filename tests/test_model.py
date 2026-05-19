import unittest
import os
import sys
import pickle
import dill
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants import (
    MODEL_EVALUATION_MODEL_NAME,
    MODEL_PUSHER_ALIAS,
)


class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """MLflow se latest production model load karo"""

        # ───────────────────────────────────────────────────────
        # Auth
        # ───────────────────────────────────────────────────────
        dagshub_token = os.getenv("DEEPSHIELD_TEST")

        if not dagshub_token:
            raise EnvironmentError(
                "DEEPSHIELD_TEST environment variable not set"
            )

        os.environ["MLFLOW_TRACKING_USERNAME"] = "kaushik-chariya"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # ───────────────────────────────────────────────────────
        # MLflow Tracking URI (duplicate removed)
        # ───────────────────────────────────────────────────────
        mlflow.set_tracking_uri(
            "https://dagshub.com/kaushik-chariya/Deep-Shield-Mail.mlflow"
        )

        print("✅ MLflow tracking URI configured")

        print("\n========== DEBUG INFO ==========")
        print("Tracking URI :", mlflow.get_tracking_uri())
        print("Model Name   :", MODEL_EVALUATION_MODEL_NAME)
        print("Alias        :", MODEL_PUSHER_ALIAS)
        print("================================\n")

        # ───────────────────────────────────────────────────────
        # Constants Validation
        # ───────────────────────────────────────────────────────
        if not MODEL_EVALUATION_MODEL_NAME:
            raise EnvironmentError(
                "MODEL_EVALUATION_MODEL_NAME is empty"
            )

        # ───────────────────────────────────────────────────────
        # MLflow Client
        # ───────────────────────────────────────────────────────
        client = mlflow.MlflowClient()

        # ───────────────────────────────────────────────────────
        # Get Latest Model Version
        # ───────────────────────────────────────────────────────
        try:
            latest_versions = client.search_model_versions(
                filter_string=f"name='{MODEL_EVALUATION_MODEL_NAME}'",
                order_by=["version_number DESC"],
                max_results=1,
            )

            if not latest_versions:
                raise Exception(
                    f"No versions found for model: "
                    f"{MODEL_EVALUATION_MODEL_NAME}"
                )

            latest_version = latest_versions[0]

            version_number = latest_version.version
            run_id         = latest_version.run_id
            source         = latest_version.source

            print(f"Latest Version : {version_number}")
            print(f"Run ID         : {run_id}")
            print(f"Source         : {source}")

            if not source:
                raise Exception(
                    "Model version source is empty — the artifact path "
                    "was not set correctly when the model was registered."
                )

        except Exception as e:
            raise Exception(f"Failed to fetch model version: {e}")

        # ───────────────────────────────────────────────────────
        # Load MLflow Model
        # ───────────────────────────────────────────────────────
        try:
            model_uri = (
                f"models:/{MODEL_EVALUATION_MODEL_NAME}/"
                f"{version_number}"
            )

            print(f"Model URI : {model_uri}")

            cls.model = mlflow.pyfunc.load_model(model_uri)

            print("✅ MLflow model loaded successfully")

        except Exception as e:
            raise Exception(f"Failed to load MLflow model: {e}")

        # ───────────────────────────────────────────────────────
        # Load Preprocessor
        # ───────────────────────────────────────────────────────
        try:
            preprocessor_uri = (
                f"runs:/{run_id}/transformers/transformers.pkl"
            )

            print(f"Preprocessor URI : {preprocessor_uri}")

            preprocessor_path = mlflow.artifacts.download_artifacts(
                artifact_uri=preprocessor_uri
            )

            with open(preprocessor_path, "rb") as f:
                cls.preprocessor = dill.load(f)

            # Debug: check what's inside
            print(f"Preprocessor type : {type(cls.preprocessor)}")
            if isinstance(cls.preprocessor, dict):
                print(f"Preprocessor keys : {list(cls.preprocessor.keys())}")

            print("✅ Preprocessor loaded successfully")

        except Exception as e:
            raise Exception(f"Failed to load preprocessor: {e}")

        # ───────────────────────────────────────────────────────
        # Load Sklearn Model (unwrapped from pyfunc)
        # ───────────────────────────────────────────────────────
        try:
            cls.sklearn_model = cls.model._model_impl

            print(f"Sklearn model type : {type(cls.sklearn_model)}")
            print("✅ Sklearn model loaded successfully")

        except Exception as e:
            raise Exception(f"Failed to load sklearn model: {e}")

    # ───────────────────────────────────────────────────────────
    # Helper Method — single place to handle preprocessor
    # ───────────────────────────────────────────────────────────

    def _transform(self, samples):
        """Preprocessor dict ya object — dono ke liye safe transform"""
        if isinstance(self.preprocessor, dict):
            vectorizer = self.preprocessor.get("vectorizer") \
                      or self.preprocessor.get("tfidf") \
                      or list(self.preprocessor.values())[0]
            transformed = vectorizer.transform(samples)
        else:
            transformed = self.preprocessor.transform(samples)

        # Convert sparse matrix to DataFrame for pyfunc compatibility
        return pd.DataFrame.sparse.from_spmatrix(transformed)

    # ───────────────────────────────────────────────────────────
    # Load Tests
    # ───────────────────────────────────────────────────────────

    def test_model_loads(self):
        """Model properly load hota hai"""
        self.assertIsNotNone(self.model)

    def test_preprocessor_loads(self):
        """Preprocessor properly load hota hai"""
        self.assertIsNotNone(self.preprocessor)

    # ───────────────────────────────────────────────────────────
    # Prediction Tests
    # ───────────────────────────────────────────────────────────

    def test_model_predicts_spam(self):
        """Spam email ko 1 predict hona chahiye"""

        sample = [
            "Congratulations! You won a free lottery. "
            "Click here to claim your reward now!"
        ]

        transformed = self._transform(sample)
        prediction  = self.model.predict(transformed)

        self.assertEqual(
            int(prediction[0]),
            1,
            "Spam email ko 1 predict hona chahiye"
        )

    def test_model_predicts_ham(self):
        """Normal email ko 0 predict hona chahiye"""

        sample = [
            "Hi team, please review the attached report "
            "before tomorrow's meeting."
        ]

        transformed = self._transform(sample)
        prediction  = self.model.predict(transformed)

        self.assertEqual(
            int(prediction[0]),
            0,
            "Normal email ko 0 predict hona chahiye"
        )

    # ───────────────────────────────────────────────────────────
    # Probability Tests
    # ───────────────────────────────────────────────────────────

    def test_spam_probability_high(self):
        """Spam probability high honi chahiye"""

        sample = [
            "Win a free iPhone now! "
            "Limited time offer! Claim your prize!"
        ]

        transformed      = self._transform(sample)
        probabilities    = self.sklearn_model.predict_proba(transformed)
        spam_probability = float(probabilities[0][1])

        self.assertGreater(
            spam_probability,
            0.7,
            (
                f"Spam probability 70% se zyada honi chahiye, "
                f"got {spam_probability:.2f}"
            )
        )

    def test_ham_probability_high(self):
        """Ham probability high honi chahiye"""

        sample = [
            "Hi team, please review the attached report "
            "before tomorrow's meeting."
        ]

        transformed      = self._transform(sample)
        probabilities    = self.sklearn_model.predict_proba(transformed)
        spam_probability = float(probabilities[0][1])

        self.assertLess(
            spam_probability,
            0.3,
            (
                f"Ham email ki spam probability "
                f"30% se kam honi chahiye, "
                f"got {spam_probability:.2f}"
            )
        )

    # ───────────────────────────────────────────────────────────
    # Batch Prediction Test
    # ───────────────────────────────────────────────────────────

    def test_model_batch_prediction(self):
        """Multiple emails ek saath predict hone chahiye"""

        samples = [
            "Congratulations! You have won a free lottery!",
            "Hi team, please review the report.",
            "Click here to claim your free prize now!",
            "Meeting is scheduled for tomorrow at 10am.",
        ]

        transformed = self._transform(samples)
        predictions = self.model.predict(transformed)

        self.assertEqual(len(predictions), 4)

        for pred in predictions:
            self.assertIn(int(pred), [0, 1])


if __name__ == "__main__":
    unittest.main()