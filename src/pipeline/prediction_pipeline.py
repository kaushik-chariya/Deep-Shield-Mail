# ═══════════════════════════════════════════════════════════════
# Prediction Pipeline
# ═══════════════════════════════════════════════════════════════

import os
import sys
import glob
import numpy as np
import pandas as pd
import pickle
import re
import dill

from scipy.sparse import hstack, csr_matrix

from constants import SCHEMA_FILE_PATH
from src.utils.logger     import logger
from src.utils.exception  import MyException
from src.utils.main_utils import read_yaml_file
from src.components.data_transformation import (
    EmailParser,
    EmailMetaFeatureExtractor,
    BodyFeatureExtractor,
    HAND_FEAT_COLS,                 # same list — single source of truth
)

import warnings
warnings.filterwarnings("ignore")


# ───────────────────────────────────────────────────────────────
# Constants — training ke saath match karo
# ───────────────────────────────────────────────────────────────

# Yeh wahi columns hain jo data_transformation.py mein drop hote hain
_DROP_HIGH_CARDINALITY = ["from_email", "to_domain"]


# ───────────────────────────────────────────────────────────────
# File helpers
# ───────────────────────────────────────────────────────────────

def _get_latest(base_dir: str, pattern: str) -> str:
    files = glob.glob(f"{base_dir}/**/{pattern}", recursive=True)
    if not files:
        raise FileNotFoundError(f"No '{pattern}' found inside '{base_dir}'")
    latest = max(files, key=os.path.getmtime)
    logger.info("📂 Latest '%s' → %s", pattern, latest)
    return latest


# ───────────────────────────────────────────────────────────────
# Header injection helper
# ───────────────────────────────────────────────────────────────

def _inject_headers_if_missing(email_text: str) -> str:
    """
    Agar user sirf plain text paste kare (bina headers ke),
    dummy headers inject karo taaki EmailParser sahi kaam kare.
    """
    header_keywords = ("from:", "to:", "subject:", "date:", "message-id:", "mime-version:")
    preview         = email_text.strip().lower()[:200]
    has_headers     = any(
        preview.startswith(kw) or f"\n{kw}" in preview
        for kw in header_keywords
    )
    if not has_headers:
        lines        = [l.strip() for l in email_text.strip().split("\n") if l.strip()]
        subject_line = lines[0][:100] if lines else "No Subject"
        logger.info("⚠️  No headers detected — injecting dummy headers")
        return (
            f"From: unknown@unknown.com\n"
            f"To: user@gmail.com\n"
            f"Subject: {subject_line}\n"
            f"Date: Mon, 17 May 2026 00:00:00 +0000\n"
            f"\n"
            f"{email_text.strip()}"
        )
    return email_text


# ───────────────────────────────────────────────────────────────
# PredictionPipeline
# ───────────────────────────────────────────────────────────────

class PredictionPipeline:
    """
    Raw email text → Spam / Not Spam

    Internally replicates the exact same steps as DataTransformation:
        EmailParser → MetaFeatures → BodyFeatures
        → drop columns
        → TF-IDF (body) + MinMaxScaler (hand-crafted)
        → hstack → MultinomialNB.predict
    """

    def __init__(self):
        try:
            # ── Load clf ───────────────────────────────────────
            model_path = _get_latest("./artifact", "model.pkl")
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info("📦 clf loaded from: %s", model_path)

            # ── Load transformers ──────────────────────────────
            # Keys: body, scaler, email_parser,
            #       meta_feature_extractor, body_feature_extractor
            transformers_path = _get_latest("./artifact", "transformers.pkl")
            with open(transformers_path, "rb") as f:
                self.transformers = dill.load(f)
            logger.info("🔧 transformers loaded from: %s", transformers_path)

            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            logger.info("✅ PredictionPipeline initialized")

        except Exception as e:
            raise MyException(e, sys)

    # ── Private ─────────────────────────────────────────────────

    def _drop_schema_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = self._schema_config.get("drop_columns", [])
        existing  = [c for c in drop_cols if c in df.columns]
        return df.drop(columns=existing) if existing else df

    def _drop_high_cardinality(self, df: pd.DataFrame) -> pd.DataFrame:
        to_drop = [c for c in _DROP_HIGH_CARDINALITY if c in df.columns]
        return df.drop(columns=to_drop) if to_drop else df

    def _build_feature_matrix(self, df: pd.DataFrame) -> csr_matrix:
        """
        Exact mirror of DataTransformation._build_feature_matrix (transform mode):
            1. preprocess_email → TF-IDF (body)
            2. HAND_FEAT_COLS   → MinMaxScaler
            3. hstack
        """
        # Step A: TF-IDF on cleaned body
        clean_body = df["body"].fillna("").apply(EmailParser.preprocess_email)
        x_body     = self.transformers["body"].transform(clean_body)
        logger.info("TF-IDF shape: %s", x_body.shape)

        # Step B: hand-crafted features → MinMaxScaler
        missing = [c for c in HAND_FEAT_COLS if c not in df.columns]
        if missing:
            logger.warning("⚠️  Missing hand-crafted columns: %s — filling with 0", missing)
            for col in missing:
                df[col] = 0

        x_hand        = df[HAND_FEAT_COLS].fillna(0).values.astype(np.float64)
        x_hand_scaled = self.transformers["scaler"].transform(x_hand)
        x_hand_sparse = csr_matrix(x_hand_scaled)
        logger.info("Hand-crafted features shape: %s", x_hand_sparse.shape)

        # Step C: combine
        X_final = hstack([x_body, x_hand_sparse])
        logger.info("Final feature matrix shape: %s", X_final.shape)
        return X_final

    # ── Public API ───────────────────────────────────────────────

    def predict(self, email_text: str) -> dict:
        """
        Parameters
        ----------
        email_text : str
            Raw email string (with or without headers).

        Returns
        -------
        dict
            {
                "label"      : "Spam" | "Not Spam",
                "prediction" : 1      | 0,
                "probability": float    # spam probability 0-1
            }
        """
        try:
            logger.info("=" * 50)
            logger.info("🔮 Prediction: STARTED")

            # Step 1: Headers inject if needed
            email_text = _inject_headers_if_missing(email_text)

            # Step 2: EmailParser — from/subject/date/to/body extract
            logger.info("[1/4] EmailParser")
            X = self.transformers["email_parser"].transform(
                pd.Series([email_text])
            )

            # Step 3: MetaFeatureExtractor — same_domain, is_weekend, to_is_generic
            logger.info("[2/4] MetaFeatureExtractor")
            X = self.transformers["meta_feature_extractor"].transform(X)

            # Step 4: BodyFeatureExtractor — caps_ratio, url_count etc.
            logger.info("[3/4] BodyFeatureExtractor")
            X = self.transformers["body_feature_extractor"].transform(X)

            # Step 5: Drop columns (same as training)
            X = self._drop_schema_columns(X)
            X = self._drop_high_cardinality(X)

            # Step 6: TF-IDF + Scaler → hstack → predict
            logger.info("[4/4] Building feature matrix & predicting")
            X_final          = self._build_feature_matrix(X)
            prediction       = self.model.predict(X_final)
            prediction_proba = self.model.predict_proba(X_final)[:, 1]

            label       = "Spam" if prediction[0] == 1 else "Not Spam"
            probability = round(float(prediction_proba[0]), 4)

            result = {
                "label"      : label,
                "prediction" : int(prediction[0]),
                "probability": probability,
            }
            logger.info("✅ Prediction result: %s", result)
            logger.info("=" * 50)
            return result

        except Exception as e:
            raise MyException(e, sys)


# ───────────────────────────────────────────────────────────────
# Smoke test
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    pipeline = PredictionPipeline()

    test_cases = [
        ("SPAM", "Congratulations! You've won a $1000 gift card. Click here to claim FREE prize NOW!"),
        ("HAM",  "Hi John, please find the meeting notes attached. Let me know if you have questions."),
        ("SPAM", "URGENT: Your bank account has been suspended. Verify your credit card details immediately."),
        ("HAM",  "Hey, are we still on for lunch tomorrow? Let me know if the time works."),
    ]

    print("\n" + "=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    passed = 0
    for expected, email in test_cases:
        result = pipeline.predict(email)
        got    = "SPAM" if result["prediction"] == 1 else "HAM"
        ok     = "✅" if got == expected else "❌"
        if got == expected:
            passed += 1
        print(f"\n{ok} Expected: {expected:4s}  Got: {got:4s}  "
              f"Prob: {result['probability']:.4f}")
        print(f"   {email[:70]}...")

    print(f"\n{'='*60}")
    print(f"Result: {passed}/{len(test_cases)} passed")
    print("=" * 60)