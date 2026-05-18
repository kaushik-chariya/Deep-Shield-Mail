# ═══════════════════════════════════════════════════════════════
# Model Pusher
# ═══════════════════════════════════════════════════════════════

import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

from constants import MODEL_EVALUATION_MODEL_NAME, MODEL_PUSHER_ALIAS
from src.utils.logger    import logger
from src.utils.exception import MyException

import warnings
warnings.filterwarnings("ignore")


class ModelPusher:

    def __init__(self):
        try:
            dagshub_token = os.getenv("DEEPSHIELD_TEST")
            if not dagshub_token:
                raise EnvironmentError(
                    "DEEPSHIELD_TEST environment variable is not set. "
                    "Please export it before running."
                )

            os.environ["MLFLOW_TRACKING_USERNAME"] = "kaushik-chariya"
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

            import dagshub
            dagshub.init(
                repo_owner="kaushik-chariya",
                repo_name ="Deep-Shield-Mail",
                mlflow    = True,
            )
            logger.info("ModelPusher: DagsHub + MLflow initialized ✅")

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_pusher(self, model_info: dict) -> None:
        """
        model_info expected keys (set by ModelEvaluation):
            push_model : bool   — True agar naya model better hai
            run_id     : str    — MLflow run ID
            new_score  : float  — naye model ka score
            best_score : float  — purane best model ka score (0.0 if first time)
        """
        try:
            logger.info("=" * 60)
            logger.info("Model Pusher: STARTED")

            # ── Guard: push_model flag check ───────────────────
            if not model_info.get('push_model', False):
                new_score  = model_info.get('new_score',  'N/A')
                best_score = model_info.get('best_score', 'N/A')
                logger.info(
                    "Model better nahi hai — Push SKIPPED "
                    "(new=%.4f  best=%.4f)",
                    new_score, best_score,
                )
                logger.info("Model Pusher: COMPLETED (skipped) ✅")
                logger.info("=" * 60)
                return

            # ── Push: register aur alias set karo ─────────────
            run_id     = model_info['run_id']
            new_score  = model_info.get('new_score',  'N/A')
            best_score = model_info.get('best_score', 0.0)
            source_uri = f"runs:/{run_id}/model"

            logger.info("New score  : %.4f", new_score)
            logger.info("Best score : %.4f", best_score)
            logger.info("Model URI  : %s",   source_uri)

            client = MlflowClient()

            # Register model (already exists toh ignore karo)
            try:
                client.create_registered_model(MODEL_EVALUATION_MODEL_NAME)
                logger.info("Registered model '%s' created", MODEL_EVALUATION_MODEL_NAME)
            except Exception:
                logger.info("Registered model '%s' already exists", MODEL_EVALUATION_MODEL_NAME)

            # New version create karo
            model_version = client.create_model_version(
                name  =MODEL_EVALUATION_MODEL_NAME,
                source=source_uri,
                run_id=run_id,
            )
            logger.info("Registered version: %s", model_version.version)

            # Champion alias set karo
            client.set_registered_model_alias(
                name   =MODEL_EVALUATION_MODEL_NAME,
                alias  =MODEL_PUSHER_ALIAS,
                version=model_version.version,
            )
            logger.info(
                "'%s' v%s → alias='%s' ✅",
                MODEL_EVALUATION_MODEL_NAME,
                model_version.version,
                MODEL_PUSHER_ALIAS,
            )

            logger.info("Model Pusher: COMPLETED ✅")
            logger.info("=" * 60)

        except Exception as e:
            raise MyException(e, sys)


# ───────────────────────────────────────────────────────────────
# Standalone run
# ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import json

    report_path = 'reports/experiment_info.json'

    try:
        with open(report_path, 'r') as f:
            model_info = json.load(f)
    except FileNotFoundError:
        logger.error("File not found: %s — pehle model evaluation run karo", report_path)
        sys.exit(1)

    logger.info("Loaded model_info: %s", model_info)

    pusher = ModelPusher()
    pusher.initiate_model_pusher(model_info=model_info)