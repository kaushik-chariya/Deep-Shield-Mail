# ═══════════════════════════════════════════════════════════════
# Data Transformation Pipeline
# ═══════════════════════════════════════════════════════════════

import sys
import os
import re
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.preprocessing           import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base                    import BaseEstimator, TransformerMixin

from constants import TARGET_COLUMN, SCHEMA_FILE_PATH, HAND_FEAT_COLS

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact,
)
from src.utils.exception   import MyException
from src.utils.logger       import logger
from src.utils.main_utils   import save_object, read_yaml_file


# ───────────────────────────────────────────────────────────────
# Regex constants
# ───────────────────────────────────────────────────────────────

HEADER_KEYS_RE = re.compile(
    r'\b(From|Return-Path|Delivered-To|Received|Message-Id|To|Subject|Date|'
    r'MIME-Version|Content-Type|Content-Transfer-Encoding|Delivery-Date|'
    r'Reply-To|Cc|Bcc|In-Reply-To|References|Importance|Thread-Index|'
    r'Thread-Topic|Organization|X-[\w-]+):\s*',
    re.IGNORECASE,
)

# Hand-crafted feature columns — notebook features + meta features
# HAND_FEAT_COLS = [
#     # ── Notebook body features ───────────────────────────────────
#     'caps_ratio',
#     'exclamation_count',
#     'url_count',
#     'dollar_count',
#     'html_flag',
#     'word_count',
#     'avg_word_length',
#     'digit_ratio',
#     'unique_word_ratio',
#     # ── Meta features (email header signals) ────────────────────
#     'same_domain',    # spam aksar alag domain se aata hai
#     'is_weekend',     # weekend pe spam zyada
#     'to_is_generic',  # noreply/admin — spam pattern
# ]


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
        body       = re.sub(r'\s+', ' ', body).strip()
        return body

    @staticmethod
    def _get_field(pattern: str, text: str) -> str:
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ''

    @staticmethod
    def preprocess_email(text: str) -> str:
        """
        NLP text cleaning — applied on raw email text before TF-IDF.
        Matches notebook: exp-Naive_Bayes.ipynb → preprocess_email()
        """
        if not isinstance(text, str):
            return ''
        text = text.lower()
        # Remove email header lines
        text = re.sub(
            r'^(from|subject|to|cc|bcc|received|content-type|mime-version|'
            r'message-id|return-path|delivered-to|x-[a-z-]+):.*$',
            '', text, flags=re.MULTILINE | re.IGNORECASE,
        )
        text = re.sub(r'https?://\S+|www\.\S+',   ' url ',   text)  # URLs → token
        text = re.sub(r'[\w.+-]+@[\w-]+\.[\w.-]+', ' email ', text)  # Emails → token
        text = re.sub(r'\$\s*\d+[\d,.]*',          ' money ', text)  # Money → token
        text = re.sub(r'\b(\+?\d[\s.-]?){7,15}\b', ' phone ', text)  # Phone → token
        text = re.sub(r'<[^>]+>',  ' ', text)                        # HTML tags
        text = re.sub(r'[^a-z\s]', ' ', text)                        # Non-alpha chars
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _parse_single(self, text: str) -> dict:
        text = str(text)
        return {
            'from'   : self._get_field(r'From:\s*(.+?)(?=\s+[\w-]+:|$)',    text),
            'subject': self._get_field(r'Subject:\s*(.+?)(?=\s+[\w-]+:|$)', text),
            'date'   : self._get_field(r'Date:\s*(.+?)(?=\s+[\w-]+:|$)',    text),
            'to'     : self._get_field(r'To:\s*(.+?)(?=\s+[\w-]+:|$)',      text),
            'body'   : self._extract_body(text),
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

        df['day_of_week'] = df['date'].str.extract(
            r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)', expand=False
        )
        df['is_weekend'] = df['day_of_week'].isin(['Sat', 'Sun']).astype(int)

        for col in ['to_email', 'to_domain', 'from_email', 'from_domain']:
            df[col] = df[col].fillna('unknown')

        logger.info("EmailMetaFeatureExtractor.transform: finished — output shape: %s", df.shape)
        return df


class BodyFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts hand-crafted features from email body.
    Feature set matches notebook: exp-Naive_Bayes.ipynb → extract_features()
    MinMaxScaler is applied later in _build_feature_matrix (required for MultinomialNB).
    """

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _parse_hour(date_str: str) -> int:
        match = re.search(r'(\d{2}):\d{2}:\d{2}', str(date_str))
        return int(match.group(1)) if match else -1

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        logger.info("BodyFeatureExtractor.transform: started — input shape: %s", X.shape)
        df  = X.copy()

        # Use raw body for hand-crafted features (before NLP cleaning) — matches notebook
        raw   = df['body'].fillna('')
        words = raw.str.split()

        # ── Notebook hand-crafted features ──────────────────────────────
        df['caps_ratio'] = raw.apply(
            lambda t: sum(c.isupper() for c in t) / max(len(t), 1)
        )
        df['exclamation_count'] = raw.str.count(r'!')
        df['url_count']         = raw.str.count(r'https?://|www\.')
        df['dollar_count']      = raw.str.count(r'\$')
        df['html_flag']         = raw.str.contains(
            r'<html|<body|<table|<td|<font', case=False, regex=True
        ).astype(int)
        df['word_count']        = words.str.len().fillna(0)
        df['avg_word_length']   = words.apply(
            lambda ws: np.mean([len(w) for w in ws])
            if isinstance(ws, list) and ws else 0
        )
        df['digit_ratio']       = raw.apply(
            lambda t: sum(c.isdigit() for c in t) / max(len(t), 1)
        )
        df['unique_word_ratio'] = words.apply(
            lambda ws: len(set(ws)) / max(len(ws), 1)
            if isinstance(ws, list) and ws else 0
        )
        # ────────────────────────────────────────────────────────────────

        logger.info(
            "BodyFeatureExtractor.transform: hand-crafted features added — %s",
            HAND_FEAT_COLS,
        )
        logger.info("BodyFeatureExtractor.transform: finished — output shape: %s", df.shape)
        return df


# ───────────────────────────────────────────────────────────────
# Save Helper
# ───────────────────────────────────────────────────────────────

def save_numpy_array(file_path: str, array: np.ndarray) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, array)
        logger.info('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error while saving data: %s', e)
        raise


# ───────────────────────────────────────────────────────────────
# DataTransformation
# ───────────────────────────────────────────────────────────────

class DataTransformation:

    def __init__(
        self,
        data_ingestion_artifact    : DataIngestionArtifact,
        data_transformation_config : DataTransformationConfig,
        data_validation_artifact   : DataValidationArtifact,
    ):
        try:
            self.data_ingestion_artifact    = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact   = data_validation_artifact
            self._schema_config             = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            self._params                    = read_yaml_file(file_path="params.yaml")
            logger.info("DataTransformation: schema loaded from '%s'", SCHEMA_FILE_PATH)
            logger.info("DataTransformation: params loaded from 'params.yaml'")
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
        """
        Returns transformers dict:
          - 'body'   : TfidfVectorizer (params from params.yaml → data_transformation.tfidf)
          - 'scaler' : MinMaxScaler    (required for MultinomialNB — keeps values >= 0)
        """
        logger.info("get_data_transformer_object: initialising sub-transformers")
        try:
            tfidf_cfg = self._params["data_transformation"]["tfidf"]

            trf_body = TfidfVectorizer(
                max_features = tfidf_cfg["max_features"],
                ngram_range  = tuple(tfidf_cfg["ngram_range"]),
                sublinear_tf = tfidf_cfg["sublinear_tf"],
                analyzer     = tfidf_cfg["analyzer"],
                stop_words   = tfidf_cfg["stop_words"],
                min_df       = tfidf_cfg["min_df"],
                max_df       = tfidf_cfg["max_df"],
                strip_accents= tfidf_cfg["strip_accents"],
            )
            scaler = MinMaxScaler()  # MultinomialNB requires non-negative features

            logger.info(
                "get_data_transformer_object: TF-IDF params — max_features=%s, "
                "ngram_range=%s, sublinear_tf=%s, min_df=%s, max_df=%s",
                tfidf_cfg["max_features"], tfidf_cfg["ngram_range"],
                tfidf_cfg["sublinear_tf"], tfidf_cfg["min_df"], tfidf_cfg["max_df"],
            )
            logger.info(
                "get_data_transformer_object: MinMaxScaler initialised for %d hand-crafted features",
                len(HAND_FEAT_COLS),
            )
            return {
                "body"  : trf_body,
                "scaler": scaler,
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
        logger.info(
            "_drop_train_duplicates: before=%d, after=%d, removed=%d",
            before, len(df_dedup), before - len(df_dedup),
        )
        return df_dedup

    def _drop_high_cardinality_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        candidates = ['from_email', 'to_domain']
        to_drop    = [c for c in candidates if c in df.columns]
        return df.drop(columns=to_drop)

    def _build_feature_matrix(self, df: pd.DataFrame, transformers: dict, fit: bool):
        """
        Builds final sparse feature matrix:
          1. preprocess_email → TfidfVectorizer on cleaned body text
          2. MinMaxScaler     on HAND_FEAT_COLS (keeps values >= 0 for MultinomialNB)
          3. hstack([tfidf_sparse, scaled_hand_features])
        """
        mode = "fit_transform" if fit else "transform"
        logger.info("_build_feature_matrix: started — mode=%s, input shape=%s", mode, df.shape)

        # ── Step A: NLP cleaning → TF-IDF ───────────────────────────────
        logger.info("_build_feature_matrix: applying preprocess_email on body text")
        clean_body = df['body'].fillna('').apply(EmailParser.preprocess_email)
        logger.info(
            "_build_feature_matrix: text cleaning done — %d documents cleaned",
            len(clean_body),
        )

        if fit:
            x_body = transformers["body"].fit_transform(clean_body)
        else:
            x_body = transformers["body"].transform(clean_body)
        logger.info("_build_feature_matrix: TF-IDF shape — %s", x_body.shape)

        # ── Step B: Hand-crafted features → MinMaxScaler ─────────────────
        missing_cols = [c for c in HAND_FEAT_COLS if c not in df.columns]
        if missing_cols:
            logger.warning(
                "_build_feature_matrix: missing hand-crafted columns — %s", missing_cols
            )

        x_hand = df[HAND_FEAT_COLS].fillna(0).values.astype(np.float64)
        if fit:
            x_hand_scaled = transformers["scaler"].fit_transform(x_hand)
        else:
            x_hand_scaled = transformers["scaler"].transform(x_hand)
        x_hand_sparse = csr_matrix(x_hand_scaled)
        logger.info(
            "_build_feature_matrix: hand-crafted features scaled — shape=%s",
            x_hand_sparse.shape,
        )

        # ── Step C: Combine ───────────────────────────────────────────────
        X_final = hstack([x_body, x_hand_sparse])
        logger.info("_build_feature_matrix: final combined shape=%s", X_final.shape)
        return X_final

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("=" * 70)
            logger.info("Data Transformation Pipeline: STARTED")
            logger.info("=" * 70)

            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Step 1: Load
            logger.info("[Step 1/10] Loading raw CSVs")
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df  = self.read_data(self.data_ingestion_artifact.test_file_path)

            # Step 2: Separate target
            logger.info("[Step 2/10] Separating target column '%s'", TARGET_COLUMN)
            y_train     = train_df[TARGET_COLUMN].reset_index(drop=True)
            y_test      = test_df[TARGET_COLUMN].reset_index(drop=True)
            X_train_raw = train_df.drop(columns=[TARGET_COLUMN]).reset_index(drop=True)
            X_test_raw  = test_df.drop(columns=[TARGET_COLUMN]).reset_index(drop=True)

            # Step 3: Parse email
            logger.info("[Step 3/10] Parsing raw email text — extracting from/subject/date/to/body")
            parser  = EmailParser()
            X_train = parser.fit_transform(X_train_raw['text'])
            X_test  = parser.transform(X_test_raw['text'])

            # Step 4: Meta features
            logger.info("[Step 4/10] Extracting meta features — domains, day_of_week, is_weekend")
            meta_extractor = EmailMetaFeatureExtractor()
            X_train = meta_extractor.fit_transform(X_train)
            X_test  = meta_extractor.transform(X_test)

            # Step 5: Body / hand-crafted features (notebook-style)
            logger.info("[Step 5/10] Extracting hand-crafted body features — %s", HAND_FEAT_COLS)
            body_extractor = BodyFeatureExtractor()
            X_train = body_extractor.fit_transform(X_train)
            X_test  = body_extractor.transform(X_test)

            # Step 6: Drop schema columns
            logger.info("[Step 6/10] Dropping schema-defined columns")
            X_train = self._drop_columns(X_train)
            X_test  = self._drop_columns(X_test)

            # Step 7: Drop high cardinality
            logger.info("[Step 7/10] Dropping high-cardinality columns")
            X_train = self._drop_high_cardinality_cols(X_train)
            X_test  = self._drop_high_cardinality_cols(X_test)

            # Step 8: Dedup train only
            logger.info("[Step 8/10] Deduplicating train set")
            before_dedup  = len(X_train)
            X_train_dedup = self._drop_train_duplicates(X_train)
            kept_indices  = X_train_dedup.index
            y_train       = y_train.iloc[kept_indices].reset_index(drop=True)
            X_train       = X_train_dedup.reset_index(drop=True)
            logger.info("[Step 8/10] before=%d, after=%d", before_dedup, len(X_train))

            assert len(X_test) == len(y_test), (
                f"X_test / y_test length mismatch: {len(X_test)} vs {len(y_test)}"
            )

            # Step 9: Build sparse matrices
            # TF-IDF (cleaned body) + MinMaxScaler (hand-crafted) → hstack
            logger.info("[Step 9/10] Building sparse feature matrices — TF-IDF + hand-crafted features")
            transformers  = self.get_data_transformer_object()
            X_train_final = self._build_feature_matrix(X_train, transformers, fit=True)
            X_test_final  = self._build_feature_matrix(X_test,  transformers, fit=False)

            assert X_train_final.shape[1] == X_test_final.shape[1], (
                f"Feature width mismatch — train: {X_train_final.shape[1]}, "
                f"test: {X_test_final.shape[1]}"
            )
            logger.info(
                "[Step 9/10] train shape=%s, test shape=%s",
                X_train_final.shape, X_test_final.shape,
            )

            # Step 10: Concatenate features + target
            logger.info("[Step 10/10] Concatenating features with target")
            y_train_sparse = csr_matrix(np.array(y_train).reshape(-1, 1))
            y_test_sparse  = csr_matrix(np.array(y_test).reshape(-1, 1))
            train_arr      = hstack([X_train_final, y_train_sparse]).toarray()
            test_arr       = hstack([X_test_final,  y_test_sparse]).toarray()
            logger.info("[Step 10/10] train=%s, test=%s", train_arr.shape, test_arr.shape)

            # Save artifacts
            logger.info("Saving transformation artifacts")
            all_transformers = {
                "body"                  : transformers["body"],
                "scaler"                : transformers["scaler"],
                "email_parser"          : parser,
                "meta_feature_extractor": meta_extractor,
                "body_feature_extractor": body_extractor,
            }

            save_object(
                self.data_transformation_config.transformed_object_file_path,
                all_transformers,
            )
            save_numpy_array(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )
            save_numpy_array(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )

            logger.info("=" * 70)
            logger.info("Data Transformation Pipeline: COMPLETED SUCCESSFULLY ✅")
            logger.info("=" * 70)

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path =self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path  =self.data_transformation_config.transformed_test_file_path,
            )
        except Exception as e:
            logger.error("Data Transformation: FAILED — %s", str(e), exc_info=True)
            raise MyException(e, sys) from e


if __name__ == "__main__":
    from src.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
    from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

    ing_cfg = DataIngestionConfig()
    val_cfg = DataValidationConfig()

    ingestion_artifact = DataIngestionArtifact(
        trained_file_path=ing_cfg.training_file_path,
        test_file_path=ing_cfg.testing_file_path,
    )
    validation_artifact = DataValidationArtifact(
        validation_status=True,
        message="OK",
        validation_report_file_path=val_cfg.validation_report_file_path,
    )

    dt = DataTransformation(
        data_ingestion_artifact=ingestion_artifact,
        data_validation_artifact=validation_artifact,
        data_transformation_config=DataTransformationConfig(),
    )
    dt.initiate_data_transformation()