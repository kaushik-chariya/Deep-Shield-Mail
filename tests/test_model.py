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

MODEL_PATH = os.path.join(ARTIFACT_DIR, MODEL_TRAINER_DIR_NAME,
                          MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)

PREPROC_PATH = os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_DIR_NAME,
                            DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                            PREPROCESSING_OBJECT_FILE_NAME)


class TestModel(unittest.TestCase):

    def test_model_file_exists(self):
        """Model file exist karta hai"""
        self.assertTrue(os.path.exists(MODEL_PATH),
                        f"Model file not found: {MODEL_PATH}")

    def test_preprocessor_file_exists(self):
        """Preprocessor file exist karta hai"""
        self.assertTrue(os.path.exists(PREPROC_PATH),
                        f"Preprocessor not found: {PREPROC_PATH}")

    def test_model_loads(self):
        """Model pickle se load hota hai"""
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        self.assertIsNotNone(model)

    def test_preprocessor_loads(self):
        """Preprocessor pickle se load hota hai"""
        with open(PREPROC_PATH, "rb") as f:
            preprocessor = pickle.load(f)
        self.assertIsNotNone(preprocessor)

    def test_model_predicts_spam(self):
        """Model spam email predict karta hai"""
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(PREPROC_PATH, "rb") as f:
            preprocessor = pickle.load(f)

        sample = ["Congratulations! You have won a free lottery. Click here to claim now!"]
        transformed = preprocessor.transform(sample)
        prediction = model.predict(transformed)

        self.assertEqual(len(prediction), 1)
        self.assertIn(int(prediction[0]), [0, 1])

    def test_model_predicts_ham(self):
        """Model normal email predict karta hai"""
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(PREPROC_PATH, "rb") as f:
            preprocessor = pickle.load(f)

        sample = ["Hi team, please review the attached report before tomorrow's meeting."]
        transformed = preprocessor.transform(sample)
        prediction = model.predict(transformed)

        self.assertEqual(len(prediction), 1)
        self.assertIn(int(prediction[0]), [0, 1])

    def test_model_predict_proba(self):
        """Model probability output deta hai"""
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(PREPROC_PATH, "rb") as f:
            preprocessor = pickle.load(f)

        sample = ["Win a free iPhone now! Limited time offer!"]
        transformed = preprocessor.transform(sample)
        proba = model.predict_proba(transformed)

        self.assertEqual(proba.shape[1], 2)
        self.assertAlmostEqual(float(sum(proba[0])), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()