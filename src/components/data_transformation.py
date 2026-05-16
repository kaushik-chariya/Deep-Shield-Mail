# ═══════════════════════════════════════════════════════════════
# Data Transformation Pipeline
# ═══════════════════════════════════════════════════════════════

import sys
import re
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix          # ✅ FIX 1: removed unused save_npz

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
from src.utils.logger    import logger               # ✅ FIX 2: was 'logging', use 'logger'

from src.utils.main_utils import (
    save_object,
    save_numpy_array_data,
    read_yaml_file
)


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
    """Parses raw email text into structured fields: from, subject, date, to, body."""

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
        # ✅ FIX 3: was "d%d" typo → fixed to "%d"
        logger.info(
            "EmailParser.transform: started — input size: %d emails", len(X)
        )

        parsed = X.apply(self._parse_single)
        df     = pd.DataFrame(list(parsed))

        null_counts  = df.isnull().sum().to_dict()
        empty_counts = {col: (df[col] == '').sum() for col in df.columns}
        body_len     = df['body'].str.len()

        logger.info("EmailParser.transform: null counts        — %s", null_counts)
        logger.info("EmailParser.transform: empty-string counts — %s", empty_counts)
        logger.info(
            "EmailParser.transform: body length — min=%d, median=%.0f, max=%d",
            body_len.min(), body_len.median(), body_len.max()
        )
        logger.info(
            "EmailParser.transform: finished — output shape: %s, columns: %s",
            df.shape, list(df.columns)
        )
        return df


class EmailMetaFeatureExtractor(BaseEstimator, TransformerMixin):
    """Derives email/domain fields and date-based features from parsed email DataFrame."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        logger.info(
            "EmailMetaFeatureExtractor.transform: started — input shape: %s", X.shape
        )

        df = X.copy()

        # ── Sender / receiver ─────────────────────────────────
        df['from_email']  = df['from'].str.extract(r'<?([\w.\+\-]+@[\w.\-]+\.\w+)>?')
        df['from_domain'] = df['from_email'].str.extract(r'@([\w.\-]+)')
        df['to_email']    = df['to'].str.extract(r'<?([\w.\+\-]+@[\w.\-]+\.\w+)>?')
        df['to_domain']   = df['to_email'].str.extract(r'@([\w.\-]+)')

        logger.info(
            "EmailMetaFeatureExtractor.transform: email extraction — "
            "from_email nulls=%d, to_email nulls=%d",
            df['from_email'].isnull().sum(),
            df['to_email'].isnull().sum()
        )
        logger.info(
            "EmailMetaFeatureExtractor.transform: unique domains — "
            "from_domain=%d, to_domain=%d",
            df['from_domain'].nunique(), df['to_domain'].nunique()
        )

        # ── Domain-level features ──────────────────────────────
        df['same_domain']   = (df['from_domain'] == df['to_domain']).astype(int)
        df['to_is_generic'] = df['to_email'].str.contains(
            r'yyyy|localhost|noreply|admin|no-reply',
            case=False, na=False
        ).astype(int)

        logger.info(
            "EmailMetaFeatureExtractor.transform: domain signals — "
            "same_domain=%d (%.1f%%), to_is_generic=%d (%.1f%%)",
            df['same_domain'].sum(),   df['same_domain'].mean()   * 100,
            df['to_is_generic'].sum(), df['to_is_generic'].mean() * 100
        )

        # ── Date features ──────────────────────────────────────
        df['day_of_week'] = df['date'].str.extract(
            r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)', expand=False
        )
        df['is_weekend'] = df['day_of_week'].isin(['Sat', 'Sun']).astype(int)

        logger.info(
            "EmailMetaFeatureExtractor.transform: day_of_week counts — %s",
            df['day_of_week'].value_counts().to_dict()
        )
        logger.info(
            "EmailMetaFeatureExtractor.transform: "
            "is_weekend=%d (%.1f%%), day_of_week nulls=%d",
            df['is_weekend'].sum(), df['is_weekend'].mean() * 100,
            df['day_of_week'].isnull().sum()
        )

        # ── Fill NAs ───────────────────────────────────────────
        for col in ['to_email', 'to_domain', 'from_email', 'from_domain']:
            n_filled = df[col].isnull().sum()
            df[col]  = df[col].fillna('unknown')
            if n_filled:
                logger.info(
                    "EmailMetaFeatureExtractor.transform: "
                    "filled %d NaN(s) with 'unknown' in '%s'",
                    n_filled, col
                )

        logger.info(
            "EmailMetaFeatureExtractor.transform: finished — output shape: %s", df.shape
        )
        return df


class BodyFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts spam-signal features from the email body and date."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        logger.info(
            "BodyFeatureExtractor.transform: started — input shape: %s", X.shape
        )

        df   = X.copy()
        body = df['body'].fillna('').str
        date = df['date'].fillna('')

        # ── Body text features ────────────────────────────────
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

        logger.info(
            "BodyFeatureExtractor.transform: keyword flags — "
            "has_link=%d, has_free=%d, has_win=%d, has_urgent=%d",
            df['has_link'].sum(), df['has_free'].sum(),
            df['has_win'].sum(),  df['has_urgent'].sum()
        )
        logger.info(
            "BodyFeatureExtractor.transform: count features — "
            "num_exclaim=%.2f, num_dollar=%.2f, num_question=%.2f, num_links=%.2f",
            df['num_exclaim'].mean(), df['num_dollar'].mean(),
            df['num_question'].mean(), df['num_links'].mean()
        )
        logger.info(
            "BodyFeatureExtractor.transform: body_len — "
            "min=%d, median=%.0f, max=%d, mean=%.0f",
            df['body_len'].min(), df['body_len'].median(),
            df['body_len'].max(), df['body_len'].mean()
        )
        logger.info(
            "BodyFeatureExtractor.transform: caps_ratio — mean=%.4f, max=%.4f",
            df['caps_ratio'].mean(), df['caps_ratio'].max()
        )

        # ── Date / time features ──────────────────────────────
        df['hour'] = date.apply(
            lambda d: int(m.group(1))
            if (m := re.search(r'(\d{2}):\d{2}:\d{2}', str(d))) else -1
        )
        n_missing_hour = (df['hour'] == -1).sum()
        logger.info(
            "BodyFeatureExtractor.transform: hour — missing: %d (%.1f%%)",
            n_missing_hour, n_missing_hour / len(df) * 100
        )

        df['is_odd_hour'] = df['hour'].apply(
            lambda h: int(h != -1 and (h < 6 or h > 22))
        )
        logger.info(
            "BodyFeatureExtractor.transform: is_odd_hour=%d (%.1f%%)",
            df['is_odd_hour'].sum(), df['is_odd_hour'].mean() * 100
        )
        logger.info(
            "BodyFeatureExtractor.transform: finished — output shape: %s", df.shape
        )
        return df


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
            logger.info(
                "DataTransformation.__init__: schema loaded from '%s'",
                SCHEMA_FILE_PATH
            )
        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────
    # Static helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        logger.info("read_data: reading '%s'", file_path)
        try:
            df = pd.read_csv(file_path)
            if "Unnamed: 0" in df.columns:
                df.drop(columns=["Unnamed: 0"], inplace=True)
                logger.info("read_data: dropped 'Unnamed: 0' index column")
            logger.info(
                "read_data: loaded — shape=%s, columns=%s",
                df.shape, list(df.columns)
            )
            return df
        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────
    # Transformer Object
    # ──────────────────────────────────────────────────────────

    def get_data_transformer_object(self) -> dict:
        """Returns dict of unfitted sub-transformers."""
        logger.info("get_data_transformer_object: initialising sub-transformers")
        try:
            trf_subject = TfidfVectorizer(
                max_features=3000, stop_words='english',
                ngram_range=(1, 2), min_df=2
            )
            trf_body = TfidfVectorizer(
                max_features=15000, stop_words='english',
                ngram_range=(1, 2), min_df=2, sublinear_tf=True
            )
            trf_from_domain = TfidfVectorizer(max_features=50)
            trf_to_email    = TfidfVectorizer(max_features=50)
            trf_day = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("ohe",     OneHotEncoder(handle_unknown='ignore'))
            ])

            logger.info(
                "get_data_transformer_object: "
                "subject(3000) | body(15000) | from_domain(50) | "
                "to_email(50) | day(OHE) — all initialised ✅"
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

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop schema-defined raw columns e.g. text, from, to, date."""
        drop_cols = self._schema_config.get('drop_columns', [])
        existing  = [c for c in drop_cols if c     in df.columns]
        missing   = [c for c in drop_cols if c not in df.columns]

        if missing:
            logger.warning("_drop_columns: schema columns not found (skipped): %s", missing)
        if existing:
            logger.info("_drop_columns: dropping: %s", existing)
            df = df.drop(columns=existing)
        else:
            logger.info("_drop_columns: nothing to drop")

        logger.info("_drop_columns: output shape: %s", df.shape)
        return df

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows on subject + body."""
        before   = len(df)
        df_dedup = df.drop_duplicates(subset=['subject', 'body'])
        removed  = before - len(df_dedup)
        logger.info(
            "_drop_duplicates: before=%d, after=%d, removed=%d (%.2f%% kept)",
            before, len(df_dedup), removed, len(df_dedup) / before * 100
        )
        return df_dedup

    def _drop_high_cardinality_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop from_email (high cardinality) and to_domain (low signal)."""
        candidates = ['from_email', 'to_domain']
        to_drop    = [c for c in candidates if c     in df.columns]
        missing    = [c for c in candidates if c not in df.columns]

        if missing:
            logger.warning(
                "_drop_high_cardinality_cols: expected columns not found: %s", missing
            )
        for col in to_drop:
            logger.info(
                "_drop_high_cardinality_cols: dropping '%s' — unique values: %d",
                col, df[col].nunique()
            )

        df = df.drop(columns=to_drop)
        logger.info(
            "_drop_high_cardinality_cols: output shape=%s, remaining=%s",
            df.shape, list(df.columns)
        )
        return df

    # ──────────────────────────────────────────────────────────
    # Build Feature Matrix
    # ──────────────────────────────────────────────────────────

    def _build_feature_matrix(
        self,
        df          : pd.DataFrame,
        transformers: dict,
        fit         : bool
    ):
        mode = "fit_transform" if fit else "transform"
        logger.info(
            "_build_feature_matrix: started — mode=%s, input shape=%s",
            mode, df.shape
        )

        num_col = self._schema_config.get('num_features', [
            'same_domain', 'to_is_generic', 'is_weekend',
            'has_link', 'num_exclaim', 'num_dollar', 'caps_ratio',
            'num_question', 'has_free', 'has_win', 'has_urgent',
            'body_len', 'num_links', 'num_digits', 'is_odd_hour'
        ])

        missing_num = [c for c in num_col if c not in df.columns]
        if missing_num:
            logger.warning(
                "_build_feature_matrix: numeric columns missing: %s", missing_num
            )

        # ── apply helpers ──────────────────────────────────────
        def apply(trf, col):
            return (
                trf.fit_transform(df[col].fillna('')) if fit
                else trf.transform(df[col].fillna(''))
            )

        def apply_day(trf, col):
            return (
                trf.fit_transform(df[[col]]) if fit
                else trf.transform(df[[col]])
            )

        # ── [1/5] Subject TF-IDF ───────────────────────────────
        logger.info("_build_feature_matrix: [1/5] subject TF-IDF")
        x_subject = apply(transformers["subject"], "subject")
        logger.info(
            "_build_feature_matrix: subject — shape=%s, nnz=%d",
            x_subject.shape, x_subject.nnz
        )

        # ── [2/5] Body TF-IDF ─────────────────────────────────
        logger.info("_build_feature_matrix: [2/5] body TF-IDF")
        x_body = apply(transformers["body"], "body")
        logger.info(
            "_build_feature_matrix: body — shape=%s, nnz=%d",
            x_body.shape, x_body.nnz
        )

        # ── [3/5] From Domain TF-IDF ──────────────────────────
        logger.info("_build_feature_matrix: [3/5] from_domain TF-IDF")
        x_from_domain = apply(transformers["from_domain"], "from_domain")
        logger.info(
            "_build_feature_matrix: from_domain — shape=%s, nnz=%d",
            x_from_domain.shape, x_from_domain.nnz
        )

        # ── [4/5] To Email TF-IDF ─────────────────────────────
        logger.info("_build_feature_matrix: [4/5] to_email TF-IDF")
        x_to_email = apply(transformers["to_email"], "to_email")
        logger.info(
            "_build_feature_matrix: to_email — shape=%s, nnz=%d",
            x_to_email.shape, x_to_email.nnz
        )

        # ── [5/5] Day OHE ─────────────────────────────────────
        logger.info("_build_feature_matrix: [5/5] day-of-week OHE")
        x_day = apply_day(transformers["day"], "day_of_week")
        logger.info(
            "_build_feature_matrix: day OHE — shape=%s", x_day.shape
        )

        # ── Numeric block ──────────────────────────────────────
        num_values = df[num_col].fillna(0).values.astype(np.float64)
        x_num      = csr_matrix(num_values)
        logger.info(
            "_build_feature_matrix: numeric block — shape=%s, nnz=%d",
            x_num.shape, x_num.nnz
        )

        # ── Stack all blocks ───────────────────────────────────
        X_final = hstack([
            x_subject, x_body, x_from_domain,
            x_to_email, x_day, x_num
        ])

        block_widths = {
            "subject"    : x_subject.shape[1],
            "body"       : x_body.shape[1],
            "from_domain": x_from_domain.shape[1],
            "to_email"   : x_to_email.shape[1],
            "day_ohe"    : x_day.shape[1],
            "numeric"    : x_num.shape[1],
        }
        logger.info(
            "_build_feature_matrix: final shape=%s, nnz=%d, "
            "density=%.4f%%, block_widths=%s",
            X_final.shape, X_final.nnz,
            X_final.nnz / (X_final.shape[0] * X_final.shape[1]) * 100,
            block_widths
        )
        return X_final

    # ──────────────────────────────────────────────────────────
    # Main Entry Point
    # ──────────────────────────────────────────────────────────

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("=" * 70)
            logger.info("Data Transformation Pipeline: STARTED")
            logger.info("=" * 70)

            # ── Validation check ───────────────────────────────
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            logger.info("[Check] Data validation passed ✅")

            # ── Step 1: Load raw CSVs ──────────────────────────
            logger.info("[Step 1/11] Loading raw CSVs")
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df  = self.read_data(self.data_ingestion_artifact.test_file_path)
            logger.info(
                "[Step 1/11] train=%s, test=%s",
                train_df.shape, test_df.shape
            )

            # ── Step 2: Separate target ────────────────────────
            logger.info("[Step 2/11] Separating target column '%s'", TARGET_COLUMN)
            y_train = train_df[TARGET_COLUMN].reset_index(drop=True)
            y_test  = test_df[TARGET_COLUMN].reset_index(drop=True)
            logger.info(
                "[Step 2/11] y_train=%s, y_test=%s",
                y_train.value_counts().to_dict(),
                y_test.value_counts().to_dict()
            )

            X_train_raw = train_df.drop(columns=[TARGET_COLUMN]).reset_index(drop=True)
            X_test_raw  = test_df.drop(columns=[TARGET_COLUMN]).reset_index(drop=True)

            # ── Step 3: Parse email text ───────────────────────
            logger.info("[Step 3/11] Parsing raw email text")
            parser  = EmailParser()
            X_train = parser.fit_transform(X_train_raw['text'])
            X_test  = parser.transform(X_test_raw['text'])
            logger.info(
                "[Step 3/11] post-parse — X_train=%s, X_test=%s",
                X_train.shape, X_test.shape
            )

            # ── Step 4: Meta features ──────────────────────────
            logger.info("[Step 4/11] Extracting meta features")
            meta_extractor = EmailMetaFeatureExtractor()
            X_train = meta_extractor.fit_transform(X_train)
            X_test  = meta_extractor.transform(X_test)
            logger.info(
                "[Step 4/11] post-meta — X_train=%s, X_test=%s",
                X_train.shape, X_test.shape
            )

            # ── Step 5: Body features ──────────────────────────
            logger.info("[Step 5/11] Extracting body features")
            body_extractor = BodyFeatureExtractor()
            X_train = body_extractor.fit_transform(X_train)
            X_test  = body_extractor.transform(X_test)
            logger.info(
                "[Step 5/11] post-body — X_train=%s, X_test=%s",
                X_train.shape, X_test.shape
            )

            # ── Step 6: Drop schema columns ────────────────────
            logger.info("[Step 6/11] Dropping schema-defined columns")
            X_train = self._drop_columns(X_train)
            X_test  = self._drop_columns(X_test)

            # ── Step 7: Drop high cardinality ──────────────────
            logger.info("[Step 7/11] Dropping high-cardinality columns")
            X_train = self._drop_high_cardinality_cols(X_train)
            X_test  = self._drop_high_cardinality_cols(X_test)
            logger.info(
                "[Step 7/11] remaining columns: %s", list(X_train.columns)
            )

            # ── Step 8: Dedup train only ───────────────────────
            # ✅ FIX 4: was resetting index BEFORE saving kept_indices
            #           so y_train.iloc[X_train.index] was always 0,1,2...
            #           which silently gave wrong y_train alignment
            logger.info(
                "[Step 8/11] Deduplicating train set (test left untouched)"
            )
            before_dedup = len(X_train)
            X_train_dedup = self._drop_duplicates(X_train)
            kept_indices  = X_train_dedup.index          # ✅ save BEFORE reset
            y_train       = y_train.iloc[kept_indices].reset_index(drop=True)
            X_train       = X_train_dedup.reset_index(drop=True)
            logger.info(
                "[Step 8/11] before=%d, after=%d, removed=%d, y_train=%s",
                before_dedup, len(X_train),
                before_dedup - len(X_train), y_train.shape
            )

            # ── Step 9: Build sparse matrices ─────────────────
            logger.info("[Step 9/11] Building sparse feature matrices")
            transformers  = self.get_data_transformer_object()
            X_train_final = self._build_feature_matrix(X_train, transformers, fit=True)
            X_test_final  = self._build_feature_matrix(X_test,  transformers, fit=False)
            logger.info(
                "[Step 9/11] train=%s, test=%s",
                X_train_final.shape, X_test_final.shape
            )

            # ── Step 10: Concatenate features + target ─────────
            logger.info("[Step 10/11] Concatenating features with target")
            train_arr = np.c_[X_train_final.toarray(), np.array(y_train)]
            test_arr  = np.c_[X_test_final.toarray(),  np.array(y_test)]
            logger.info(
                "[Step 10/11] final arrays — train=%s, test=%s",
                train_arr.shape, test_arr.shape
            )

            # ── Step 11: Save artifacts ────────────────────────
            logger.info("[Step 11/11] Saving transformation artifacts")

            save_object(
                self.data_transformation_config.transformed_object_file_path,
                transformers
            )
            logger.info(
                "[Step 11/11] preprocessor saved → 💾 '%s'",
                self.data_transformation_config.transformed_object_file_path
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            logger.info(
                "[Step 11/11] train array saved → 📝 '%s'",
                self.data_transformation_config.transformed_train_file_path
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )
            logger.info(
                "[Step 11/11] test array saved → 📝 '%s'",
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
            logger.error(
                "Data Transformation: FAILED — %s", str(e), exc_info=True
            )
            raise MyException(e, sys) from e