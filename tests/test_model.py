import unittest
import os
import sys
import dill
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants import (
    MODEL_EVALUATION_MODEL_NAME,
    MODEL_PUSHER_ALIAS,
)

from src.components.data_transformation import (
    EmailParser,
    HAND_FEAT_COLS,
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

        # ───────────────────────────────────────────────────────
        # MLflow Tracking URI — pehle define karo
        # ───────────────────────────────────────────────────────
        tracking_uri = (
            f"https://kaushik-chariya:{dagshub_token}"
            "@dagshub.com/kaushik-chariya/Deep-Shield-Mail.mlflow"
        )

        # Saare env vars set karo
        os.environ["MLFLOW_TRACKING_USERNAME"] = "kaushik-chariya"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri  # ← FIX
        os.environ["DAGSHUB_USER_TOKEN"]       = dagshub_token

        import dagshub                                           # ← ADD
        dagshub.init(                                            # ← ADD
        repo_owner="kaushik-chariya",                        # ← ADD
        repo_name="Deep-Shield-Mail",                        # ← ADD
        mlflow=True,                                         # ← ADD
)                                                        # ← ADD

        mlflow.set_tracking_uri(tracking_uri)


        mlflow.set_tracking_uri(tracking_uri)

        print("✅ MLflow tracking URI configured")

        print("\n========== DEBUG INFO ==========")
        print("Tracking URI :", "https://dagshub.com/kaushik-chariya/Deep-Shield-Mail.mlflow")
        print("Model Name   :", MODEL_EVALUATION_MODEL_NAME)
        print("Alias        :", MODEL_PUSHER_ALIAS)
        print("================================\n")

        # ───────────────────────────────────────────────────────
        # MLflow Client — usi tracking URI se banao
        # ───────────────────────────────────────────────────────
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)

        # ───────────────────────────────────────────────────────
        # Get Latest Model Version
        # ───────────────────────────────────────────────────────
        latest_version = client.get_model_version_by_alias(
            MODEL_EVALUATION_MODEL_NAME,
            MODEL_PUSHER_ALIAS
        )

        if not latest_version:
            raise Exception(
                f"No versions found for model: "
                f"{MODEL_EVALUATION_MODEL_NAME}"
            )

        version_number = latest_version.version
        run_id         = latest_version.run_id

        print(f"Latest Version : {version_number}")
        print(f"Run ID         : {run_id}")

        # ───────────────────────────────────────────────────────
        # Load MLflow Model
        # ───────────────────────────────────────────────────────
        model_uri = (
            f"models:/{MODEL_EVALUATION_MODEL_NAME}/"
            f"{version_number}"
        )

        print(f"Model URI : {model_uri}")

        cls.model = mlflow.pyfunc.load_model(model_uri)

        print("✅ MLflow model loaded successfully")

        # ───────────────────────────────────────────────────────
        # Load Preprocessor
        # ───────────────────────────────────────────────────────
        preprocessor_uri = (
            f"runs:/{run_id}/transformers/transformers.pkl"
        )

        print(f"Preprocessor URI : {preprocessor_uri}")

        preprocessor_path = mlflow.artifacts.download_artifacts(
            artifact_uri=preprocessor_uri
        )

        with open(preprocessor_path, "rb") as f:
            cls.preprocessor = dill.load(f)

        print(f"Preprocessor type : {type(cls.preprocessor)}")

        if isinstance(cls.preprocessor, dict):
            print(f"Preprocessor keys : {list(cls.preprocessor.keys())}")

        print("✅ Preprocessor loaded successfully")

        # ───────────────────────────────────────────────────────
        # Load sklearn model
        # ───────────────────────────────────────────────────────
        cls.sklearn_model = cls.model._model_impl

        print(f"Sklearn model type : {type(cls.sklearn_model)}")
        print("✅ Sklearn model loaded successfully")

    # ───────────────────────────────────────────────────────────
    # FULL TRANSFORMATION PIPELINE
    # ───────────────────────────────────────────────────────────

    def _transform(self, samples):

        # Step 1: EmailParser
        X = self.preprocessor["email_parser"].transform(
            pd.Series(samples)
        )

        # Step 2: Meta Features
        X = self.preprocessor["meta_feature_extractor"].transform(X)

        # Step 3: Body Features
        X = self.preprocessor["body_feature_extractor"].transform(X)

        # Step 4: Clean body
        clean_body = X["body"].fillna("").apply(
            EmailParser.preprocess_email
        )

        # Step 5: TF-IDF
        x_body = self.preprocessor["body"].transform(clean_body)

        # Step 6: Hand-crafted features
        x_hand = (
            X[HAND_FEAT_COLS]
            .fillna(0)
            .values
            .astype(np.float64)
        )

        # Step 7: Scaling
        x_hand_scaled = self.preprocessor["scaler"].transform(
            x_hand
        )

        x_hand_sparse = csr_matrix(x_hand_scaled)

        # Step 8: Combine features
        transformed = hstack([x_body, x_hand_sparse])

        print(f"Final transformed shape : {transformed.shape}")

        # Convert to pandas sparse DataFrame for MLflow pyfunc
        return pd.DataFrame.sparse.from_spmatrix(transformed)

    # ───────────────────────────────────────────────────────────
    # Load Tests
    # ───────────────────────────────────────────────────────────

    def test_model_loads(self):
        self.assertIsNotNone(self.model)

    def test_preprocessor_loads(self):
        self.assertIsNotNone(self.preprocessor)

    # ───────────────────────────────────────────────────────────
    # Prediction Tests
    # ───────────────────────────────────────────────────────────

    def test_model_predicts_spam(self):

        sample = [
            "Congratulations! You won a free lottery. "
            "Click here to claim your reward now!"
        ]

        transformed = self._transform(sample)

        prediction = self.model.predict(transformed)

        self.assertEqual(
            int(prediction[0]),
            1
        )

    def test_model_predicts_ham(self):

        sample = [
            "Hi team, please review the attached report "
            "before tomorrow's meeting."
        ]

        transformed = self._transform(sample)

        prediction = self.model.predict(transformed)

        self.assertEqual(
            int(prediction[0]),
            0
        )

    # ───────────────────────────────────────────────────────────
    # Probability Tests
    # ───────────────────────────────────────────────────────────

    def test_spam_probability_high(self):

        sample = [
            "Win a free iPhone now! "
            "Limited time offer! Claim your prize!"
        ]

        transformed = self._transform(sample)

        probabilities = self.sklearn_model.predict_proba(
            transformed
        )

        spam_probability = float(probabilities[0][1])

        self.assertGreater(
            spam_probability,
            0.7
        )

    def test_ham_probability_high(self):

        sample = [
            "Hi team, please review the attached report "
            "before tomorrow's meeting."
        ]

        transformed = self._transform(sample)

        probabilities = self.sklearn_model.predict_proba(
            transformed
        )

        spam_probability = float(probabilities[0][1])

        self.assertLess(
            spam_probability,
            0.3
        )

    # ───────────────────────────────────────────────────────────
    # Batch Prediction Test
    # ───────────────────────────────────────────────────────────

    def test_model_batch_prediction(self):

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