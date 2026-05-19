# ═══════════════════════════════════════════════════════════════
# Model Evaluation
# ═══════════════════════════════════════════════════════════════

import os
import sys
import glob
import json
import pickle
import dill

import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from constants import (
    MODEL_EVALUATION_EXPERIMENT_NAME,
    MODEL_EVALUATION_MODEL_NAME,
    MODEL_PUSHER_ALIAS,
)
from src.utils.logger    import logger
from src.utils.exception import MyException

import warnings
warnings.filterwarnings("ignore")


class ModelEvaluation:

    def __init__(self):
        try:
            dagshub_token = os.getenv("DEEPSHIELD_TEST")
            if not dagshub_token:
                raise EnvironmentError(
                    "DEEPSHIELD_TEST environment variable is not set."
                )

            os.environ["MLFLOW_TRACKING_USERNAME"] = "kaushik-chariya"
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

            mlflow.set_tracking_uri(
                "https://dagshub.com/kaushik-chariya/Deep-Shield-Mail.mlflow"
            )
            logger.info("✅ ModelEvaluation: DagsHub + MLflow initialized")

        except Exception as e:
            raise MyException(e, sys)

    # ── Private helpers ─────────────────────────────────────────

    def _get_latest_file(self, base_dir: str, pattern: str) -> str:
        files = glob.glob(f"{base_dir}/**/{pattern}", recursive=True)
        if not files:
            raise FileNotFoundError(
                f"No '{pattern}' found inside '{base_dir}'"
            )
        latest = max(files, key=os.path.getmtime)
        logger.info("📂 Latest '%s' found at: %s", pattern, latest)
        return latest

    def _load_model(self, file_path: str):
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        logger.info("📦 Model (clf) loaded from: %s", file_path)
        return model

    def _load_transformers(self, file_path: str) -> dict:
        with open(file_path, "rb") as f:
            transformers = dill.load(f)
        logger.info("🔧 Transformers loaded from: %s", file_path)
        return transformers

    def _load_test_data(self, file_path: str) -> np.ndarray:
        data = np.load(file_path)
        logger.info("📊 Test data loaded — shape: %s", data.shape)
        return data

    def _evaluate(
        self,
        clf    : object,
        X_test : np.ndarray,
        y_test : np.ndarray,
    ) -> dict:
        y_pred       = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        return {
            "accuracy" : round(accuracy_score( y_test, y_pred),       4),
            "f1"       : round(f1_score(       y_test, y_pred),       4),
            "precision": round(precision_score(y_test, y_pred),       4),
            "recall"   : round(recall_score(   y_test, y_pred),       4),
            "auc"      : round(roc_auc_score(  y_test, y_pred_proba), 4),
        }

    def _get_champion_accuracy(self, model_name: str) -> float:
        try:
            client           = MlflowClient()
            champion_version = client.get_model_version_by_alias(
                name =model_name,
                alias=MODEL_PUSHER_ALIAS,
            )
            prev_run = client.get_run(champion_version.run_id)
            best_acc = prev_run.data.metrics.get("accuracy", 0.0)
            logger.info(
                "🏆 Champion (alias='%s') accuracy: %.4f",
                MODEL_PUSHER_ALIAS, best_acc,
            )
            return best_acc

        except Exception:
            logger.info(
                "⚠️  Koi champion nahi mila (alias='%s') — "
                "pehli baar hai, push hoga",
                MODEL_PUSHER_ALIAS,
            )
            return 0.0

    # ── Main ────────────────────────────────────────────────────

    def initiate_model_evaluation(self) -> dict:
        try:
            logger.info("=" * 60)
            logger.info("🚀 Model Evaluation: STARTED")
            logger.info("=" * 60)

            mlflow.set_experiment(MODEL_EVALUATION_EXPERIMENT_NAME)

            with mlflow.start_run() as run:

                # ── Step 1: Load artifacts ─────────────────────
                model_path        = self._get_latest_file("./artifact", "model.pkl")
                transformers_path = self._get_latest_file("./artifact", "transformers.pkl")
                test_data_path    = self._get_latest_file("./artifact", "test.npy")

                clf          = self._load_model(model_path)
                transformers = self._load_transformers(transformers_path)
                test_arr     = self._load_test_data(test_data_path)

                X_test = test_arr[:, :-1]
                y_test = test_arr[:, -1]
                logger.info("📐 X_test=%s | y_test=%s", X_test.shape, y_test.shape)

                # ── Step 2: Evaluate ───────────────────────────
                metrics = self._evaluate(clf, X_test, y_test)
                logger.info("📈 Accuracy  : %.4f", metrics["accuracy"])
                logger.info("📈 F1        : %.4f", metrics["f1"])
                logger.info("📈 Precision : %.4f", metrics["precision"])
                logger.info("📈 Recall    : %.4f", metrics["recall"])
                logger.info("📈 AUC       : %.4f", metrics["auc"])

                # ── Step 3: Log metrics ────────────────────────
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)

                # ── Step 4: Log params ─────────────────────────
                if hasattr(clf, "get_params"):
                    for k, v in clf.get_params().items():
                        mlflow.log_param(k, v)

                # ── Step 5: Log model ──────────────────────────
                # FIX: name="model" → artifact_path="model"
                # name="" → MLflow registry mein path='' store hota tha → CI crash
                # artifact_path="model" → sahi path store hota hai → load hoga ✅
                input_example = X_test[:5]
                signature     = infer_signature(X_test, clf.predict(X_test))

                mlflow.sklearn.log_model(
                    sk_model      = clf,
                    name = "model",        # ← FIXED (was: name="model")
                    signature     = signature,
                    input_example = input_example,
                )
                logger.info("✅ clf logged to MLflow (artifact_path='model')")

                # ── Step 6: Log transformers.pkl ───────────────
                mlflow.log_artifact(
                    local_path   =transformers_path,
                    artifact_path="transformers",
                )
                logger.info(
                    "✅ transformers.pkl logged to MLflow "
                    "(artifact_path='transformers/transformers.pkl')"
                )

                # ── Step 7: Save metrics report ────────────────
                os.makedirs("reports", exist_ok=True)
                with open("reports/metrics.json", "w") as f:
                    json.dump(metrics, f, indent=4)
                mlflow.log_artifact("reports/metrics.json")
                logger.info("💾 Metrics saved → reports/metrics.json")

                # ── Step 8: Compare with champion ─────────────
                best_accuracy = self._get_champion_accuracy(MODEL_EVALUATION_MODEL_NAME)
                new_accuracy  = metrics["accuracy"]
                should_push   = new_accuracy > best_accuracy

                if should_push:
                    logger.info(
                        "🚀 New model BETTER (%.4f > %.4f) — PUSH hoga",
                        new_accuracy, best_accuracy,
                    )
                else:
                    logger.info(
                        "⏭️  New model NOT better (%.4f <= %.4f) — SKIP",
                        new_accuracy, best_accuracy,
                    )

                # ── Step 9: Save experiment_info.json ─────────
                model_info = {
                    "run_id"            : run.info.run_id,
                    "model_path"        : "model",
                    "transformers_path" : "transformers/transformers.pkl",
                    "push_model"        : should_push,
                    "new_score"         : new_accuracy,
                    "best_score"        : best_accuracy,
                }
                with open("reports/experiment_info.json", "w") as f:
                    json.dump(model_info, f, indent=4)
                logger.info(
                    "💾 Experiment info saved → reports/experiment_info.json"
                )

                logger.info("=" * 60)
                logger.info("✅ Model Evaluation: COMPLETED SUCCESSFULLY")
                logger.info("=" * 60)

                return model_info

        except Exception as e:
            logger.error(
                "❌ Model Evaluation: FAILED — %s", str(e), exc_info=True
            )
            raise MyException(e, sys)


# ───────────────────────────────────────────────────────────────
# Standalone run
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluator = ModelEvaluation()
    evaluator.initiate_model_evaluation()