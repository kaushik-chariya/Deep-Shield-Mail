import sys
import re
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix, save_npz

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
from src.utils.logger import logging

from src.utils.main_utils import (
    save_object,
    save_numpy_array_data,
    read_yaml_file
)


# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

HEADER_KEYS_RE = re.compile(
    r'\b(From|Return-Path|Delivered-To|Received|Message-Id|To|Subject|Date|'
    r'MIME-Version|Content-Type|Content-Transfer-Encoding|Delivery-Date|'
    r'Reply-To|Cc|Bcc|In-Reply-To|References|Importance|Thread-Index|'
    r'Thread-Topic|Organization|X-[\w-]+):\s*',
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Custom Transformers (sklearn-compatible)
# ---------------------------------------------------------------------------

class EmailParser(BaseEstimator, TransformerMixin):
    """
    Parses raw email text into structured fields:
    from, subject, date, to, body.
    """

    @staticmethod
    def _extract_body(text: str) -> str:
        matches = list(HEADER_KEYS_RE.finditer(text))
        if not matches:
            return text
        last_match  = matches[-1]
        after_last  = text[last_match.end():]
        value_end   = re.search(r'\s+[A-Z][^:]{10,}', after_last)
        body        = after_last[value_end.start():].strip() if value_end else after_last.strip()
        body        = HEADER_KEYS_RE.sub(' ', body)
        body        = re.sub(r'<[^>]+>', ' ', body)
        body        = re.sub(r'http\S+',  ' ', body)
        body        = re.sub(r'\s+',      ' ', body).strip()
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
        logging.info("EmailParser.transform: started — input size: %d emails", len(X))

        parsed = X.apply(self._parse_single)
        df     = pd.DataFrame(list(parsed))

        # — field-level null counts
        null_counts = df.isnull().sum().to_dict()
        logging.info("EmailParser.transform: null counts per field — %s", null_counts)

        # — empty-string counts (fields found but blank)
        empty_counts = {col: (df[col] == '').sum() for col in df.columns}
        logging.info("EmailParser.transform: empty-string counts — %s", empty_counts)

        # — body length stats
        body_len = df['body'].str.len()
        logging.info(
            "EmailParser.transform: body length — "
            "min=%d, median=%.0f, max=%d",
            body_len.min(), body_len.median(), body_len.max()
        )

        logging.info(
            "EmailParser.transform: finished — output shape: %s, columns: %s",
            df.shape, list(df.columns)
        )
        return df


class EmailMetaFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Derives email/domain fields and date-based features
    from the parsed email DataFrame.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        logging.info(
            "EmailMetaFeatureExtractor.transform: started — input shape 📩 : %s",
            X.shape
        )

        df = X.copy()

        # --- Sender / receiver fields ---
        df['from_email']  = df['from'].str.extract(r'<?([\w.\+\-]+@[\w.\-]+\.\w+)>?')
        df['from_domain'] = df['from_email'].str.extract(r'@([\w.\-]+)')
        df['to_email']    = df['to'].str.extract(r'<?([\w.\+\-]+@[\w.\-]+\.\w+)>?')
        df['to_domain']   = df['to_email'].str.extract(r'@([\w.\-]+)')

        logging.info(
            "EmailMetaFeatureExtractor.transform: email extraction — "
            "from_email nulls=%d, to_email nulls=%d",
            df['from_email'].isnull().sum(),
            df['to_email'].isnull().sum()
        )
        logging.info(
            "EmailMetaFeatureExtractor.transform: unique domains — "
            "from_domain=%d, to_domain=%d",
            df['from_domain'].nunique(),
            df['to_domain'].nunique()
        )

        # --- Domain-level features ---
        df['same_domain']   = (df['from_domain'] == df['to_domain']).astype(int)
        df['to_is_generic'] = df['to_email'].str.contains(
            r'yyyy|localhost|noreply|admin|no-reply',
            case=False, na=False
        ).astype(int)

        logging.info(
            "EmailMetaFeatureExtractor.transform: domain signals — "
            "same_domain=%d (%.1f%%), to_is_generic=%d (%.1f%%)",
            df['same_domain'].sum(),   df['same_domain'].mean()   * 100,
            df['to_is_generic'].sum(), df['to_is_generic'].mean() * 100
        )

        # --- Date features ---
        df['day_of_week'] = df['date'].str.extract(
            r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)', expand=False
        )
        df['is_weekend'] = df['day_of_week'].isin(['Sat', 'Sun']).astype(int)

        logging.info(
            "EmailMetaFeatureExtractor.transform: day_of_week value counts — %s",
            df['day_of_week'].value_counts().to_dict()
        )
        logging.info(
            "EmailMetaFeatureExtractor.transform: "
            "is_weekend=%d (%.1f%%), day_of_week nulls=%d",
            df['is_weekend'].sum(), df['is_weekend'].mean() * 100,
            df['day_of_week'].isnull().sum()
        )

        # --- Fill NAs for high-cardinality string cols ---
        for col in ['to_email', 'to_domain', 'from_email', 'from_domain']:
            n_filled = df[col].isnull().sum()
            df[col]  = df[col].fillna('unknown')
            if n_filled:
                logging.info(
                    "EmailMetaFeatureExtractor.transform: "
                    "filled %d NaN(s) with 'unknown' in column '%s'",
                    n_filled, col
                )

        logging.info(
            "EmailMetaFeatureExtractor.transform: finished — output shape: %s",
            df.shape
        )
        return df


class BodyFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts spam-signal features from the email body and subject.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        logging.info(
            "BodyFeatureExtractor.transform: started — input shape: %s",
            X.shape
        )

        df   = X.copy()
        body = df['body'].fillna('').str
        date = df['date'].fillna('')

        # --- Body text features ---
        df['has_link']     = body.contains('http', case=False).astype(int)
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
        df['num_links']    = body.count('http')
        df['num_digits']   = body.count(r'\d')

        logging.info(
            "BodyFeatureExtractor.transform: keyword flags — "
            "has_link=%d, has_free=%d, has_win=%d, has_urgent=%d",
            df['has_link'].sum(), df['has_free'].sum(),
            df['has_win'].sum(),  df['has_urgent'].sum()
        )
        logging.info(
            "BodyFeatureExtractor.transform: count features — "
            "num_exclaim mean=%.2f, num_dollar mean=%.2f, "
            "num_question mean=%.2f, num_links mean=%.2f",
            df['num_exclaim'].mean(),  df['num_dollar'].mean(),
            df['num_question'].mean(), df['num_links'].mean()
        )
        logging.info(
            "BodyFeatureExtractor.transform: body_len — "
            "min=%d, median=%.0f, max=%d, mean=%.0f",
            df['body_len'].min(), df['body_len'].median(),
            df['body_len'].max(), df['body_len'].mean()
        )
        logging.info(
            "BodyFeatureExtractor.transform: "
            "caps_ratio — mean=%.4f, max=%.4f",
            df['caps_ratio'].mean(), df['caps_ratio'].max()
        )

        # --- Date / time features ---
        df['hour'] = date.apply(
            lambda d: int(m.group(1))
            if (m := re.search(r'(\d{2}):\d{2}:\d{2}', str(d))) else -1
        )
        n_missing_hour = (df['hour'] == -1).sum()
        logging.info(
            "BodyFeatureExtractor.transform: hour extraction — "
            "missing (hour=-1): %d (%.1f%%)",
            n_missing_hour, n_missing_hour / len(df) * 100
        )

        df['is_odd_hour'] = df['hour'].apply(
            lambda h: int(h != -1 and (h < 6 or h > 22))
        )
        logging.info(
            "BodyFeatureExtractor.transform: "
            "is_odd_hour=%d (%.1f%%)",
            df['is_odd_hour'].sum(), df['is_odd_hour'].mean() * 100
        )

        logging.info(
            "BodyFeatureExtractor.transform: finished — output shape: %s",
            df.shape
        )
        return df


# ---------------------------------------------------------------------------
# DataTransformation
# ---------------------------------------------------------------------------

class DataTransformation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact
    ):
        try:
            self.data_ingestion_artifact      = data_ingestion_artifact
            self.data_transformation_config   = data_transformation_config
            self.data_validation_artifact     = data_validation_artifact
            self._schema_config               = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            logging.info(
                "DataTransformation.__init__: schema loaded from '%s'",
                SCHEMA_FILE_PATH
            )
        except Exception as e:
            raise MyException(e, sys)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        logging.info("DataTransformation.read_data: reading file '%s'", file_path)
        try:
            df = pd.read_csv(file_path)
            if "Unnamed: 0" in df.columns:
                df.drop(columns=["Unnamed: 0"], inplace=True)
                logging.info(
                    "DataTransformation.read_data: dropped 'Unnamed: 0' index column"
                )
            logging.info(
                "DataTransformation.read_data: loaded — shape: %s, columns: %s",
                df.shape, list(df.columns)
            )
            return df
        except Exception as e:
            raise MyException(e, sys)

    # ------------------------------------------------------------------
    # Preprocessing object
    # ------------------------------------------------------------------

    def get_data_transformer_object(self) -> dict:
        """
        Returns a dict of fitted transformers that together build
        the final sparse feature matrix:

            X_final = hstack([
                subject_tfidf,
                body_tfidf,
                from_domain_tfidf,
                to_email_tfidf,
                day_ohe,
                numeric_csr
            ])
        """
        logging.info("get_data_transformer_object: initialising sub-transformers")

        try:
            trf_subject = TfidfVectorizer(
                max_features=3000, stop_words='english',
                ngram_range=(1, 2), min_df=2
            )
            logging.info(
                "get_data_transformer_object: subject TfidfVectorizer — "
                "max_features=3000, ngram_range=(1,2), min_df=2"
            )

            trf_body = TfidfVectorizer(
                max_features=15000, stop_words='english',
                ngram_range=(1, 2), min_df=2, sublinear_tf=True
            )
            logging.info(
                "get_data_transformer_object: body TfidfVectorizer — "
                "max_features=15000, ngram_range=(1,2), sublinear_tf=True"
            )

            trf_from_domain = TfidfVectorizer(max_features=50)
            logging.info(
                "get_data_transformer_object: from_domain TfidfVectorizer — max_features=50"
            )

            trf_to_email = TfidfVectorizer(max_features=50)
            logging.info(
                "get_data_transformer_object: to_email TfidfVectorizer — max_features=50"
            )

            trf_day = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("ohe",     OneHotEncoder(handle_unknown='ignore'))
            ])
            logging.info(
                "get_data_transformer_object: day Pipeline — "
                "SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')"
            )

            logging.info(
                "get_data_transformer_object: all sub-transformers initialised successfully"
            )

            return {
                "subject"    : trf_subject,
                "body"       : trf_body,
                "from_domain": trf_from_domain,
                "to_email"   : trf_to_email,
                "day"        : trf_day,
            }

        except Exception as e:
            raise MyException(e, sys) from e

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns listed in schema (e.g. 'text', 'from', 'to', 'date')."""
        logging.info("_drop_columns: reading drop list from schema config")
        drop_cols = self._schema_config.get('drop_columns', [])
        existing  = [c for c in drop_cols if c in df.columns]
        missing   = [c for c in drop_cols if c not in df.columns]

        if missing:
            logging.warning(
                "_drop_columns: schema-listed columns not found (skipped): %s", missing
            )
        if existing:
            logging.info("_drop_columns: dropping columns: %s", existing)
            df = df.drop(columns=existing)
        else:
            logging.info("_drop_columns: no columns to drop — DataFrame unchanged")

        logging.info("_drop_columns: output shape: %s", df.shape)
        return df

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows based on subject + body."""
        before = len(df)
        logging.info(
            "_drop_duplicates: checking for duplicates on ['subject', 'body'] — "
            "rows before: %d",
            before
        )
        df_dedup = df.drop_duplicates(subset=['subject', 'body'])
        removed  = before - len(df_dedup)
        logging.info(
            "_drop_duplicates: removed %d duplicate row(s) — rows after: %d (%.2f%% kept)",
            removed, len(df_dedup), len(df_dedup) / before * 100
        )
        return df_dedup

    def _drop_high_cardinality_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop from_email (high cardinality) and to_domain (low signal)."""
        candidates = ['from_email', 'to_domain']
        to_drop    = [c for c in candidates if c in df.columns]
        missing    = [c for c in candidates if c not in df.columns]

        if missing:
            logging.warning(
                "_drop_high_cardinality_cols: expected columns not found: %s", missing
            )

        for col in to_drop:
            logging.info(
                "_drop_high_cardinality_cols: dropping '%s' — unique values: %d",
                col, df[col].nunique()
            )

        df = df.drop(columns=to_drop)
        logging.info(
            "_drop_high_cardinality_cols: output shape: %s, remaining columns: %s",
            df.shape, list(df.columns)
        )
        return df

    # ------------------------------------------------------------------
    # Build feature matrix
    # ------------------------------------------------------------------

    def _build_feature_matrix(
        self,
        df: pd.DataFrame,
        transformers: dict,
        fit: bool
    ):
        """
        Applies all transformers and stacks into a single sparse matrix.

        Parameters
        ----------
        df           : input DataFrame (after all feature engineering)
        transformers : dict of sub-transformers
        fit          : True  → fit_transform  |  False → transform only
        """
        mode = "fit_transform" if fit else "transform"
        logging.info(
            "_build_feature_matrix: started — mode=%s, input shape: %s",
            mode, df.shape
        )

        num_col = self._schema_config.get('num_features', [
            'same_domain', 'to_is_generic', 'is_weekend',
            'has_link', 'num_exclaim', 'num_dollar', 'caps_ratio',
            'num_question', 'has_free', 'has_win', 'has_urgent',
            'body_len', 'num_links', 'num_digits', 'is_odd_hour'
        ])
        logging.info(
            "_build_feature_matrix: numeric feature columns (%d): %s",
            len(num_col), num_col
        )

        # — check for missing numeric columns before proceeding
        missing_num = [c for c in num_col if c not in df.columns]
        if missing_num:
            logging.warning(
                "_build_feature_matrix: numeric columns missing from DataFrame: %s",
                missing_num
            )

        apply = lambda trf, col: (
            trf.fit_transform(df[col].fillna(''))
            if fit else
            trf.transform(df[col].fillna(''))
        )
        apply_day = lambda trf, col: (
            trf.fit_transform(df[[col]])
            if fit else
            trf.transform(df[[col]])
        )

        logging.info("_build_feature_matrix: [1/5] applying subject TF-IDF")
        x_subject = apply(transformers["subject"], "subject")
        logging.info(
            "_build_feature_matrix: subject TF-IDF shape: %s, nnz: %d",
            x_subject.shape, x_subject.nnz
        )

        logging.info("_build_feature_matrix: [2/5] applying body TF-IDF")
        x_body = apply(transformers["body"], "body")
        logging.info(
            "_build_feature_matrix: body TF-IDF shape: %s, nnz: %d",
            x_body.shape, x_body.nnz
        )

        logging.info("_build_feature_matrix: [3/5] applying from_domain TF-IDF")
        x_from_domain = apply(transformers["from_domain"], "from_domain")
        logging.info(
            "_build_feature_matrix: from_domain TF-IDF shape: %s, nnz: %d",
            x_from_domain.shape, x_from_domain.nnz
        )

        logging.info("_build_feature_matrix: [4/5] applying to_email TF-IDF")
        x_to_email = apply(transformers["to_email"], "to_email")
        logging.info(
            "_build_feature_matrix: to_email TF-IDF shape: %s, nnz: %d",
            x_to_email.shape, x_to_email.nnz
        )

        logging.info("_build_feature_matrix: [5/5] applying day-of-week OHE pipeline")
        x_day = apply_day(transformers["day"], "day_of_week")
        logging.info(
            "_build_feature_matrix: day OHE shape: %s",
            x_day.shape
        )

        # — numeric block
        num_values = df[num_col].fillna(0).values.astype(np.float64)
        nan_count  = np.isnan(num_values).sum()
        if nan_count:
            logging.warning(
                "_build_feature_matrix: %d NaN(s) found in numeric block before fillna "
                "(already handled — this should be 0)", nan_count
            )
        x_num = csr_matrix(num_values)
        logging.info(
            "_build_feature_matrix: numeric block shape: %s, nnz: %d",
            x_num.shape, x_num.nnz
        )

        # — stack all blocks
        X_final = hstack([
            x_subject, x_body, x_from_domain,
            x_to_email, x_day, x_num
        ])
        logging.info(
            "_build_feature_matrix: final sparse matrix — "
            "shape: %s, nnz: %d, density: %.4f%%",
            X_final.shape, X_final.nnz,
            X_final.nnz / (X_final.shape[0] * X_final.shape[1]) * 100
        )

        block_widths = {
            "subject"    : x_subject.shape[1],
            "body"       : x_body.shape[1],
            "from_domain": x_from_domain.shape[1],
            "to_email"   : x_to_email.shape[1],
            "day_ohe"    : x_day.shape[1],
            "numeric"    : x_num.shape[1],
        }
        logging.info(
            "_build_feature_matrix: column-block widths — %s  (total: %d)",
            block_widths, X_final.shape[1]
        )

        return X_final

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Full transformation pipeline:
          raw text → parse → meta features → body features
          → dedup (train only) → drop columns → TF-IDF + OHE + numeric hstack
          → save artifacts

        Data leakage controls
        ─────────────────────
        • _drop_duplicates : train only (test dedup would leak row counts)
        • TfidfVectorizer  : fit on train → transform test  (no leakage)
        • OHE / Imputer    : fit on train → transform test  (no leakage)
        • EmailParser / MetaExtractor / BodyExtractor :
              purely deterministic (regex / arithmetic) — no fit state,
              safe to call independently on train and test
        """
        try:
            logging.info("=" * 70)
            logging.info("Data Transformation Pipeline: STARTED")
            logging.info("=" * 70)

            if not self.data_validation_artifact.validation_status:
                logging.error(
                    "initiate_data_transformation: validation failed — %s",
                    self.data_validation_artifact.message
                )
                raise Exception(self.data_validation_artifact.message)

            logging.info(
                "initiate_data_transformation: data validation check passed"
            )

            # ── 1. Load raw data ─────────────────────────────────────────
            logging.info("initiate_data_transformation: [Step 1] Loading raw data")
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df  = self.read_data(self.data_ingestion_artifact.test_file_path)
            logging.info(
                "initiate_data_transformation: train shape=%s, test shape=%s",
                train_df.shape, test_df.shape
            )

            # ── 2. Separate target BEFORE any transformation ─────────────
            logging.info(
                "initiate_data_transformation: [Step 2] Separating target column '%s'",
                TARGET_COLUMN
            )
            y_train = train_df[TARGET_COLUMN].reset_index(drop=True)
            y_test  = test_df[TARGET_COLUMN].reset_index(drop=True)
            logging.info(
                "initiate_data_transformation: y_train distribution — %s",
                y_train.value_counts().to_dict()
            )
            logging.info(
                "initiate_data_transformation: y_test distribution — %s",
                y_test.value_counts().to_dict()
            )

            X_train_raw = train_df.drop(columns=[TARGET_COLUMN]).reset_index(drop=True)
            X_test_raw  = test_df.drop(columns=[TARGET_COLUMN]).reset_index(drop=True)
            logging.info(
                "initiate_data_transformation: X_train_raw=%s, X_test_raw=%s",
                X_train_raw.shape, X_test_raw.shape
            )

            # ── 3. Parse email text ──────────────────────────────────────
            logging.info(
                "initiate_data_transformation: [Step 3] Parsing raw email text"
            )
            parser  = EmailParser()
            X_train = parser.fit_transform(X_train_raw['text'])   # fit is no-op
            X_test  = parser.transform(X_test_raw['text'])
            logging.info(
                "initiate_data_transformation: post-parse — "
                "X_train=%s, X_test=%s",
                X_train.shape, X_test.shape
            )

            # ── 4. Extract meta features ─────────────────────────────────
            logging.info(
                "initiate_data_transformation: [Step 4] Extracting meta features"
            )
            meta_extractor = EmailMetaFeatureExtractor()
            X_train = meta_extractor.fit_transform(X_train)       # fit is no-op
            X_test  = meta_extractor.transform(X_test)
            logging.info(
                "initiate_data_transformation: post-meta — "
                "X_train=%s, X_test=%s",
                X_train.shape, X_test.shape
            )

            # ── 5. Extract body features ─────────────────────────────────
            logging.info(
                "initiate_data_transformation: [Step 5] Extracting body features"
            )
            body_extractor = BodyFeatureExtractor()
            X_train = body_extractor.fit_transform(X_train)       # fit is no-op
            X_test  = body_extractor.transform(X_test)
            logging.info(
                "initiate_data_transformation: post-body — "
                "X_train=%s, X_test=%s",
                X_train.shape, X_test.shape
            )

            # ── 6. Drop schema-defined raw columns ───────────────────────
            logging.info(
                "initiate_data_transformation: [Step 6] Dropping schema-defined columns"
            )
            X_train = self._drop_columns(X_train)
            X_test  = self._drop_columns(X_test)

            # ── 7. Drop high-cardinality / low-signal columns ─────────────
            logging.info(
                "initiate_data_transformation: [Step 7] "
                "Dropping high-cardinality/low-signal columns"
            )
            X_train = self._drop_high_cardinality_cols(X_train)
            X_test  = self._drop_high_cardinality_cols(X_test)
            logging.info(
                "initiate_data_transformation: post-column-drop — "
                "X_train=%s, X_test=%s, columns=%s",
                X_train.shape, X_test.shape, list(X_train.columns)
            )

            # ── 8. Dedup train only ───────────────────────────────────────
            logging.info(
                "initiate_data_transformation: [Step 8] "
                "Deduplicating train set (test left untouched)"
            )
            before_dedup = len(X_train)
            X_train = self._drop_duplicates(X_train).reset_index(drop=True)
            y_train = y_train.iloc[X_train.index].reset_index(drop=True)
            X_train = X_train.reset_index(drop=True)
            logging.info(
                "initiate_data_transformation: dedup — "
                "rows before=%d, after=%d, removed=%d, "
                "y_train shape=%s",
                before_dedup, len(X_train), before_dedup - len(X_train),
                y_train.shape
            )

            # ── 9. Build sparse feature matrices ──────────────────────────
            logging.info(
                "initiate_data_transformation: [Step 9] Building sparse feature matrices"
            )
            transformers  = self.get_data_transformer_object()
            X_train_final = self._build_feature_matrix(X_train, transformers, fit=True)
            X_test_final  = self._build_feature_matrix(X_test,  transformers, fit=False)
            logging.info(
                "initiate_data_transformation: sparse matrices — "
                "train=%s, test=%s",
                X_train_final.shape, X_test_final.shape
            )

            # ── 10. Concatenate features + target ─────────────────────────
            logging.info(
                "initiate_data_transformation: [Step 10] "
                "Concatenating sparse features with target column"
            )
            train_arr = np.c_[X_train_final.toarray(), np.array(y_train)]
            test_arr  = np.c_[X_test_final.toarray(),  np.array(y_test)]
            logging.info(
                "initiate_data_transformation: final dense arrays — ✔️"
                "train=%s, test=%s",
                train_arr.shape, test_arr.shape
            )

            # ── 11. Save artifacts ────────────────────────────────────────
            logging.info(
                "initiate_data_transformation: [Step 11] Saving transformation artifacts"
            )
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                transformers
            )
            logging.info(
                "initiate_data_transformation: preprocessor saved → 💾'%s'",
                self.data_transformation_config.transformed_object_file_path
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            logging.info(
                "initiate_data_transformation: train array saved → 📝'%s'",
                self.data_transformation_config.transformed_train_file_path
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )
            logging.info(
                "initiate_data_transformation: test array saved → 📝'%s'",
                self.data_transformation_config.transformed_test_file_path
            )

            logging.info("=" * 70)
            logging.info("Data Transformation Pipeline: COMPLETED SUCCESSFULLY...✅")
            logging.info("=" * 70)

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path =self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path  =self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            logging.error(
                "initiate_data_transformation: FAILED with error 🤷🏻‍♂️— %s", str(e),
                exc_info=True
            )
            raise MyException(e, sys) from e