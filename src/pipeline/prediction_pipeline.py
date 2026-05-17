import os
import sys
import glob
import dill as pickle
import re
import numpy as np
import pandas as pd
from src.utils.logger import logger

from scipy.sparse import hstack, csr_matrix
from constants import SCHEMA_FILE_PATH
from src.utils.logger    import logger
from src.utils.exception import MyException
from src.utils.main_utils import read_yaml_file
from src.components.data_transformation import EmailParser, EmailMetaFeatureExtractor, BodyFeatureExtractor

# ── Same constants as data_transformation.py ─────────────────────────────────
DEFAULT_NUM_FEATURES = [
    'same_domain', 'to_is_generic', 'is_weekend',
    'has_link', 'num_exclaim', 'num_dollar', 'caps_ratio',
    'num_question', 'has_free', 'has_win', 'has_urgent',
    'body_len', 'num_links', 'num_digits', 'is_odd_hour'
]
# ─────────────────────────────────────────────────────────────────────────────


def get_latest_model_path(base_dir: str = './artifact') -> str:
    """Automatically find the latest model.pkl from the artifact directory."""
    model_files = glob.glob(f'{base_dir}/**/model.pkl', recursive=True)
    if not model_files:
        raise FileNotFoundError(f"No model.pkl found inside '{base_dir}' directory")
    latest = max(model_files, key=os.path.getmtime)
    logger.info(f'Latest model found at: {latest}')
    return latest


def get_latest_preprocessor_path(base_dir: str = './artifact') -> str:
    """Automatically find the latest preprocessing.pkl from the artifact directory."""
    preprocessor_files = glob.glob(f'{base_dir}/**/preprocessing.pkl', recursive=True)
    if not preprocessor_files:
        raise FileNotFoundError(f"No preprocessing.pkl found inside '{base_dir}' directory")
    latest = max(preprocessor_files, key=os.path.getmtime)
    logger.info(f'Latest preprocessor found at: {latest}')
    return latest


def _inject_headers_if_missing(email_text: str) -> str:
    """
    Agar email mein headers missing hain (From, To, Subject, Date)
    toh automatically neutral headers inject karta hai.
    Yeh ensure karta hai ki EmailParser sahi se kaam kare
    even jab user sirf body paste kare.
    """
    header_keywords = ("from:", "to:", "subject:", "date:", "message-id:", "mime-version:")
    first_lines = email_text.strip().lower()[:200]

    has_headers = any(first_lines.startswith(kw) or f"\n{kw}" in first_lines for kw in header_keywords)

    if not has_headers:
        # Subject ke liye pehli non-empty line use karo
        lines = [l.strip() for l in email_text.strip().split('\n') if l.strip()]
        subject_line = lines[0][:100] if lines else "No Subject"

        logger.info("No headers detected — injecting default headers for parsing")

        injected = (
            f"From: unknown@unknown.com\n"
            f"To: user@gmail.com\n"
            f"Subject: {subject_line}\n"
            f"Date: Mon, 17 May 2026 00:00:00 +0000\n"
            f"\n"
            f"{email_text.strip()}"
        )
        return injected

    return email_text


class PredictionPipeline:

    def __init__(self):
        try:
            # ✅ Load latest model
            model_path = get_latest_model_path('./artifact')
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

            # ✅ Load latest all_transformers dict
            preprocessor_path = get_latest_preprocessor_path('./artifact')
            with open(preprocessor_path, 'rb') as f:
                self.transformers = pickle.load(f)

            # ✅ Load schema for num_features + drop_columns
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

            logger.info("PredictionPipeline initialized ✅")

        except Exception as e:
            raise MyException(e, sys)

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Same as DataTransformation._drop_columns"""
        drop_cols = self._schema_config.get('drop_columns', [])
        existing  = [c for c in drop_cols if c in df.columns]
        if existing:
            df = df.drop(columns=existing)
        return df

    def _drop_high_cardinality_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Same as DataTransformation._drop_high_cardinality_cols"""
        candidates = ['from_email', 'to_domain']
        to_drop    = [c for c in candidates if c in df.columns]
        return df.drop(columns=to_drop)

    def _build_feature_matrix(self, df: pd.DataFrame) -> csr_matrix:
        """
        Same as DataTransformation._build_feature_matrix but fit=False always.
        Uses saved transformers from preprocessing.pkl.
        """
        num_col = self._schema_config.get('num_features', DEFAULT_NUM_FEATURES)

        x_subject     = self.transformers["subject"].transform(df["subject"].fillna(''))
        x_body        = self.transformers["body"].transform(df["body"].fillna(''))
        x_from_domain = self.transformers["from_domain"].transform(df["from_domain"].fillna(''))
        x_to_email    = self.transformers["to_email"].transform(df["to_email"].fillna(''))
        x_day         = self.transformers["day"].transform(df[["day_of_week"]])

        x_num   = csr_matrix(df[num_col].fillna(0).values.astype(np.float64))
        X_final = hstack([x_subject, x_body, x_from_domain, x_to_email, x_day, x_num])

        logger.info(f"Feature matrix shape: {X_final.shape}")
        return X_final

    def predict(self, email_text: str) -> dict:
        """
        Takes raw email text as input and returns prediction.
        Automatically injects headers if missing.

        Args:
            email_text: Raw email string (with or without headers)

        Returns:
            dict with label and probability
        """
        try:
            import pandas as pd
            logger.info("=" * 50)
            logger.info("Prediction: STARTED")

            # ── Step 0: Inject headers if missing ────────────
            email_text = _inject_headers_if_missing(email_text)

            # ── Step 1: Parse email ───────────────────────────
            logger.info("[Step 1/5] Parsing email text")
            parser  = self.transformers["email_parser"]
            X       = parser.transform(pd.Series([email_text]))

            # ── Step 2: Meta features ─────────────────────────
            logger.info("[Step 2/5] Extracting meta features")
            meta_extractor = self.transformers["meta_feature_extractor"]
            X = meta_extractor.transform(X)

            # ── Step 3: Body features ─────────────────────────
            logger.info("[Step 3/5] Extracting body features")
            body_extractor = self.transformers["body_feature_extractor"]
            X = body_extractor.transform(X)

            # ── Step 4: Drop columns ──────────────────────────
            logger.info("[Step 4/5] Dropping columns")
            X = self._drop_columns(X)
            X = self._drop_high_cardinality_cols(X)

            # ── Step 5: Build feature matrix & Predict ────────
            logger.info("[Step 5/5] Building feature matrix & predicting")
            X_final = self._build_feature_matrix(X)

            prediction       = self.model.predict(X_final)
            prediction_proba = self.model.predict_proba(X_final)[:, 1]

            label       = "Spam" if prediction[0] == 1 else "Not Spam"
            probability = round(float(prediction_proba[0]), 4)

            result = {
                "label"      : label,
                "prediction" : int(prediction[0]),
                "probability": probability
            }

            logger.info(f"Prediction result: {result}")
            logger.info("Prediction: COMPLETED ✅")
            return result

        except Exception as e:
            raise MyException(e, sys)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    sample_email = "Congratulations! You've won a $1000 gift card. Click here to claim now!"

    pipeline = PredictionPipeline()
    result   = pipeline.predict(sample_email)

    print(f"\n📧 Email      : {sample_email}")
    print(f"🔍 Label      : {result['label']}")
    print(f"📊 Probability: {result['probability']}")