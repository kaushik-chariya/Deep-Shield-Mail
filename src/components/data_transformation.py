# ═══════════════════════════════════════════════════════════════
# Data Transformation Pipeline
# ═══════════════════════════════════════════════════════════════

import sys
import os
import re
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from constants import TARGET_COLUMN, SCHEMA_FILE_PATH

from src.entity.config_entity import DataTransformationConfig

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact
)

from src.utils.exception import MyException
from src.utils.logger    import logger

from src.utils.main_utils import (
    save_object,
    read_yaml_file
)


# ───────────────────────────────────────────────────────────────
# Module-level constants
# ───────────────────────────────────────────────────────────────

DEFAULT_NUM_FEATURES = [
    'same_domain', 'to_is_generic', 'is_weekend',
    'has_link', 'num_exclaim', 'num_dollar', 'caps_ratio',
    'num_question', 'has_free', 'has_win', 'has_urgent',
    'body_len', 'num_links', 'num_digits', 'is_odd_hour'
]

# ───────────────────────────────────────────────────────────────
# Regex constants
# ───────────────────────────────────────────────────────────────

HEADER_KEYS_RE = re.compile(
    r'\b(From|Return-Path|Delivered-To|Received|Message-Id|To|Subject|Date|'
    r'MIME-Version|Content-Type|Content-Transfer-Encoding|Delivery-Date|'
    r'Reply-To|Cc|Bcc|In-Reply-To|References|Importance|Thread-Index|'
    r'Thread-Topic|Organization|X-[\w-]+):\s*',
    re.IGNORECASE
)


# ───────────────────────────────────────────────────────────────
# Custom Transformers
# ───────────────────────────────────────────────────────────────

class EmailParser(BaseEstimator, TransformerMixin):

    @staticmethod
    def _extract_body(text: str) -> str:
        matches    = list(HEADER_KEYS_RE.finditer(text))
        if not matches:
            return text
        last_match = matches[-1]
        after_last = text[last_match.end():]
        value_end  = re.search(r'\s+[A-Z][^:]{10,}', after_last)
        body       = after_last[value_end.start():].strip() if value_end else after_last.strip()
        body       = HEADER_KEYS_RE.sub(' ', body)
        body       = re.sub(r'<[^>]+>', ' ', body)
        body       = re.sub(r'http\S+',  ' ', body)
        body       = re.sub(r'\s+',      ' ', body).strip()
        return body

    @staticmethod
    def _get_field(pattern: str, text: str) -> str:
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ''

    def _parse_single(self, text: str) -> dict:
        text = str(text)
        return {
            'from'   : self._get_field(r'From:\s*(.+?)(?=\s+[\w-]+:|$)',    text),
            'subject': self._get_field(r'Subject:\s*(.+?)(?=\s+[\w-]+:|$)', text),
            'date'   : self._get_field(r'Date:\s*(.+?)(?=\s+[\w-]+:|$)',    text),
            'to'     : self._get_field(r'To:\s*(.+?)(?=\s+[\w-]+:|$)',      text),
            'body'   : self._extract_body(text)
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        logger.info("EmailParser.transform: started — input size: %d emails", len(X))
        parsed = X.apply(self._parse_single)
        df     = pd.DataFrame(list(parsed))
        logger.info("EmailParser.transform: finished — output shape: %s", df.shape)
        return df


class EmailMetaFeatureExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        logger.info("EmailMetaFeatureExtractor.transform: started — input shape: %s", X.shape)
        df = X.copy()
        df['from_email']  = df['from'].str.extract(r'<?([\w.\+\-]+@[\w.\-]+\.\w+)>?')
        df['from_domain'] = df['from_email'].str.extract(r'@([\w.\-]+)')
        df['to_email']    = df['to'].str.extract(r'<?([\w.\+\-]+@[\w.\-]+\.\w+)>?')
        df['to_domain']   = df['to_email'].str.extract(r'@([\w.\-]+)')
        df['same_domain']   = (df['from_domain'] == df['to_domain']).astype(int)
        df['to_is_generic'] = df['to_email'].str.contains(
            r'yyyy|localhost|noreply|admin|no-reply', case=False, na=False
        ).astype(int)
        df['day_of_week'] = df['date'].str.extract(r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)', expand=False)
        df['is_weekend']  = df['day_of_week'].isin(['Sat', 'Sun']).astype(int)
        for col in ['to_email', 'to_domain', 'from_email', 'from_domain']:
            df[col] = df[col].fillna('unknown')
        logger.info("EmailMetaFeatureExtractor.transform: finished — output shape: %s", df.shape)
        return df


class BodyFeatureExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _parse_hour(date_str: str) -> int:
        match = re.search(r'(\d{2}):\d{2}:\d{2}', str(date_str))
        return int(match.group(1)) if match else -1

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        logger.info("BodyFeatureExtractor.transform: started — input shape: %s", X.shape)
        df   = X.copy()
        body = df['body'].fillna('').str
        date = df['date'].fillna('')
        df['has_link']     = body.contains(r'https?://', case=False).astype(int)
        df['num_exclaim']  = body.count('!')
        df['num_dollar']   = body.count(r'\$')
        df['caps_ratio']   = df['body'].fillna('').apply(
            lambda t: sum(1 for c in t if c.isupper()) / (len(t) + 1)
        )
        df['num_question'] = body.count(r'\?')
        df['has_free']     = body.contains('free',            case=False).astype(int)
        df['has_win']      = body.contains(r'win|winner',     case=False).astype(int)
        df['has_urgent']   = body.contains(r'urgent|act now', case=False).astype(int)
        df['body_len']     = body.len()
        df['num_links']    = body.count(r'https?://')
        df['num_digits']   = body.count(r'\d')
        df['hour']         = date.apply(self._parse_hour)
        df['is_odd_hour']  = df['hour'].apply(lambda h: int(h != -1 and (h < 6 or h > 22)))
        logger.info("BodyFeatureExtractor.transform: finished — output shape: %s", df.shape)
        return df


# ───────────────────────────────────────────────────────────────
# Save Helper  (same pattern as feature_engineering.py → save_data)
# ───────────────────────────────────────────────────────────────

def save_numpy_array(file_path: str, array: np.ndarray) -> None:
    """Save a numpy array to disk — mirrors save_data() in feature_engineering.py."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, array)
        logger.info('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


# ───────────────────────────────────────────────────────────────
# DataTransformation
# ───────────────────────────────────────────────────────────────

class DataTransformation:

    def __init__(
        self,
        data_ingestion_artifact    : DataIngestionArtifact,
        data_transformation_config : DataTransformationConfig,
        data_validation_artifact   : DataValidationArtifact
    ):
        try:
            self.data_ingestion_artifact    = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact   = data_validation_artifact
            self._schema_config             = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            logger.info("DataTransformation.__init__: schema loaded from '%s'", SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        logger.info("read_data: reading '%s'", file_path)
        try:
            df = pd.read_csv(file_path)
            if "Unnamed: 0" in df.columns:
                df.drop(columns=["Unnamed: 0"], inplace=True)
            logger.info("read_data: loaded — shape=%s", df.shape)
            return df
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> dict:
        logger.info("get_data_transformer_object: initialising sub-transformers")
        try:
            trf_subject     = TfidfVectorizer(max_features=3000,  stop_words='english', ngram_range=(1, 2), min_df=2)
            trf_body        = TfidfVectorizer(max_features=15000, stop_words='english', ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            trf_from_domain = TfidfVectorizer(max_features=50)
            trf_to_email    = TfidfVectorizer(max_features=50)
            trf_day = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("ohe",     OneHotEncoder(handle_unknown='ignore'))
            ])
            return {
                "subject"    : trf_subject,
                "body"       : trf_body,
                "from_domain": trf_from_domain,
                "to_email"   : trf_to_email,
                "day"        : trf_day,
            }
        except Exception as e:
            raise MyException(e, sys) from e

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = self._schema_config.get('drop_columns', [])
        existing  = [c for c in drop_cols if c in df.columns]
        if existing:
            df = df.drop(columns=existing)
        return df

    def _drop_train_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before   = len(df)
        df_dedup = df.drop_duplicates(subset=['subject', 'body'])
        logger.info("_drop_train_duplicates: before=%d, after=%d, removed=%d",
                    before, len(df_dedup), before - len(df_dedup))
        return df_dedup

    def _drop_high_cardinality_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        candidates = ['from_email', 'to_domain']
        to_drop    = [c for c in candidates if c in df.columns]
        return df.drop(columns=to_drop)

    def _build_feature_matrix(self, df: pd.DataFrame, transformers: dict, fit: bool):
        mode = "fit_transform" if fit else "transform"
        logger.info("_build_feature_matrix: started — mode=%s, input shape=%s", mode, df.shape)

        num_col = self._schema_config.get('num_features', DEFAULT_NUM_FEATURES)

        def _fit_or_transform(trf, col):
            return trf.fit_transform(df[col].fillna('')) if fit else trf.transform(df[col].fillna(''))

        def _fit_or_transform_2d(trf, col):
            return trf.fit_transform(df[[col]]) if fit else trf.transform(df[[col]])

        x_subject     = _fit_or_transform(transformers["subject"],     "subject")
        x_body        = _fit_or_transform(transformers["body"],        "body")
        x_from_domain = _fit_or_transform(transformers["from_domain"], "from_domain")
        x_to_email    = _fit_or_transform(transformers["to_email"],    "to_email")
        x_day         = _fit_or_transform_2d(transformers["day"],      "day_of_week")

        x_num   = csr_matrix(df[num_col].fillna(0).values.astype(np.float64))
        X_final = hstack([x_subject, x_body, x_from_domain, x_to_email, x_day, x_num])

        logger.info("_build_feature_matrix: final shape=%s", X_final.shape)
        return X_final

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("=" * 70)
            logger.info("Data Transformation Pipeline: STARTED")
            logger.info("=" * 70)

            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # ── Step 1: Load ───────────────────────────────────
            logger.info("[Step 1/11] Loading raw CSVs")
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df  = self.read_data(self.data_ingestion_artifact.test_file_path)

            # ── Step 2: Separate target ────────────────────────
            logger.info("[Step 2/11] Separating target column '%s'", TARGET_COLUMN)
            y_train     = train_df[TARGET_COLUMN].reset_index(drop=True)
            y_test      = test_df[TARGET_COLUMN].reset_index(drop=True)
            X_train_raw = train_df.drop(columns=[TARGET_COLUMN]).reset_index(drop=True)
            X_test_raw  = test_df.drop(columns=[TARGET_COLUMN]).reset_index(drop=True)

            # ── Step 3: Parse email ────────────────────────────
            logger.info("[Step 3/11] Parsing raw email text")
            parser  = EmailParser()
            X_train = parser.fit_transform(X_train_raw['text'])
            X_test  = parser.transform(X_test_raw['text'])

            # ── Step 4: Meta features ──────────────────────────
            logger.info("[Step 4/11] Extracting meta features")
            meta_extractor = EmailMetaFeatureExtractor()
            X_train = meta_extractor.fit_transform(X_train)
            X_test  = meta_extractor.transform(X_test)

            # ── Step 5: Body features ──────────────────────────
            logger.info("[Step 5/11] Extracting body features")
            body_extractor = BodyFeatureExtractor()
            X_train = body_extractor.fit_transform(X_train)
            X_test  = body_extractor.transform(X_test)

            # ── Step 6: Drop schema columns ────────────────────
            logger.info("[Step 6/11] Dropping schema-defined columns")
            X_train = self._drop_columns(X_train)
            X_test  = self._drop_columns(X_test)

            # ── Step 7: Drop high cardinality ──────────────────
            logger.info("[Step 7/11] Dropping high-cardinality columns")
            X_train = self._drop_high_cardinality_cols(X_train)
            X_test  = self._drop_high_cardinality_cols(X_test)

            # ── Step 8: Dedup train only ───────────────────────
            logger.info("[Step 8/11] Deduplicating train set")
            before_dedup  = len(X_train)
            X_train_dedup = self._drop_train_duplicates(X_train)
            kept_indices  = X_train_dedup.index
            y_train       = y_train.iloc[kept_indices].reset_index(drop=True)
            X_train       = X_train_dedup.reset_index(drop=True)
            logger.info("[Step 8/11] before=%d, after=%d", before_dedup, len(X_train))

            assert len(X_test) == len(y_test), (
                f"X_test / y_test length mismatch: {len(X_test)} vs {len(y_test)}"
            )

            # ── Step 9: Build sparse matrices ──────────────────
            logger.info("[Step 9/11] Building sparse feature matrices")
            transformers  = self.get_data_transformer_object()
            X_train_final = self._build_feature_matrix(X_train, transformers, fit=True)
            X_test_final  = self._build_feature_matrix(X_test,  transformers, fit=False)

            assert X_train_final.shape[1] == X_test_final.shape[1], (
                f"Feature width mismatch — train: {X_train_final.shape[1]}, "
                f"test: {X_test_final.shape[1]}"
            )

            # ── Step 10: Concatenate features + target ─────────
            logger.info("[Step 10/11] Concatenating features with target")
            y_train_sparse = csr_matrix(np.array(y_train).reshape(-1, 1))
            y_test_sparse  = csr_matrix(np.array(y_test).reshape(-1, 1))

            train_arr = hstack([X_train_final, y_train_sparse]).toarray()
            test_arr  = hstack([X_test_final,  y_test_sparse]).toarray()
            logger.info("[Step 10/11] train=%s, test=%s", train_arr.shape, test_arr.shape)

            # ── Step 11: Save artifacts ────────────────────────
            # Same pattern as feature_engineering.py → save_data()
            logger.info("[Step 11/11] Saving transformation artifacts")

            all_transformers = {
                **transformers,
                "email_parser"          : parser,
                "meta_feature_extractor": meta_extractor,
                "body_feature_extractor": body_extractor,
            }

            save_object(
                self.data_transformation_config.transformed_object_file_path,
                all_transformers
            )
            logger.info(
                "[Step 11/11] preprocessor saved → '%s'",
                self.data_transformation_config.transformed_object_file_path
            )

            save_numpy_array(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            logger.info(
                "[Step 11/11] train array saved → '%s'",
                self.data_transformation_config.transformed_train_file_path
            )

            save_numpy_array(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )
            logger.info(
                "[Step 11/11] test array saved → '%s'",
                self.data_transformation_config.transformed_test_file_path
            )

            logger.info("=" * 70)
            logger.info("Data Transformation Pipeline: COMPLETED SUCCESSFULLY ✅")
            logger.info("=" * 70)

            return DataTransformationArtifact(
                transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path  = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path   = self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            logger.error("Data Transformation: FAILED — %s", str(e), exc_info=True)
            raise MyException(e, sys) from e