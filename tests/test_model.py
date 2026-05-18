import unittest
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants import (
    ARTIFACT_DIR,
    MODEL_TRAINER_DIR_NAME,
    MODEL_TRAINER_TRAINED_MODEL_DIR,
    MODEL_FILE_NAME,
    DATA_TRANSFORMATION_DIR_NAME,
    DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
    PREPROCESSING_OBJECT_FILE_NAME,
)

MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    MODEL_TRAINER_DIR_NAME,
    MODEL_TRAINER_TRAINED_MODEL_DIR,
    MODEL_FILE_NAME,
)

PREPROC_PATH = os.path.join(
    ARTIFACT_DIR,
    DATA_TRANSFORMATION_DIR_NAME,
    DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
    PREPROCESSING_OBJECT_FILE_NAME,
)


class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Model aur preprocessor ek baar load karo"""
        with open(MODEL_PATH, "rb") as f:
            cls.model = pickle.load(f)
        with open(PREPROC_PATH, "rb") as f:
            cls.preprocessor = pickle.load(f)

    # ── File existence tests ────────────────────────────────────

    def test_model_file_exists(self):
        """Model file exist karta hai"""
        self.assertTrue(
            os.path.exists(MODEL_PATH),
            f"Model file not found: {MODEL_PATH}"
        )

    def test_preprocessor_file_exists(self):
        """Preprocessor file exist karta hai"""
        self.assertTrue(
            os.path.exists(PREPROC_PATH),
            f"Preprocessor not found: {PREPROC_PATH}"
        )

    # ── Load tests ──────────────────────────────────────────────

    def test_model_loads(self):
        """Model pickle se load hota hai"""
        self.assertIsNotNone(self.model)

    def test_preprocessor_loads(self):
        """Preprocessor pickle se load hota hai"""
        self.assertIsNotNone(self.preprocessor)

    # ── Prediction tests ────────────────────────────────────────

    def test_model_predicts_spam(self):
        """Model spam email ko 1 predict karta hai"""
        sample = ["Congratulations! You have won a free lottery. Click here to claim now!"]
        transformed = self.preprocessor.transform(sample)
        prediction = self.model.predict(transformed)
        self.assertEqual(len(prediction), 1)
        self.assertEqual(
            int(prediction[0]), 1,
            "Spam email ko 1 predict hona chahiye"
        )

    def test_model_predicts_ham(self):
        """Model normal email ko 0 predict karta hai"""
        sample = ["Hi team, please review the attached report before tomorrow's meeting."]
        transformed = self.preprocessor.transform(sample)
        prediction = self.model.predict(transformed)
        self.assertEqual(len(prediction), 1)
        self.assertEqual(
            int(prediction[0]), 0,
            "Normal email ko 0 predict hona chahiye"
        )

    # ── Probability tests ───────────────────────────────────────

    def test_model_predict_proba(self):
        """Model probability output deta hai jo 1.0 sum kare"""
        sample = ["Win a free iPhone now! Limited time offer!"]
        transformed = self.preprocessor.transform(sample)
        proba = self.model.predict_proba(transformed)
        self.assertEqual(proba.shape[1], 2)
        self.assertAlmostEqual(
            float(sum(proba[0])), 1.0, places=5,
            msg="Probabilities ka sum 1.0 hona chahiye"
        )

    def test_spam_probability_high(self):
        """Spam email ki probability 0.7 se zyada honi chahiye"""
        sample = ["Win a free iPhone now! Limited time offer! Claim your prize!"]
        transformed = self.preprocessor.transform(sample)
        proba = self.model.predict_proba(transformed)
        self.assertGreater(
            float(proba[0][1]), 0.7,
            "Spam probability 70% se zyada honi chahiye"
        )

    def test_ham_probability_high(self):
        """Normal email ki ham probability 0.7 se zyada honi chahiye"""
        sample = ["Hi team, please review the attached report before tomorrow's meeting."]
        transformed = self.preprocessor.transform(sample)
        proba = self.model.predict_proba(transformed)
        self.assertGreater(
            float(proba[0][0]), 0.7,
            "Ham probability 70% se zyada honi chahiye"
        )

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