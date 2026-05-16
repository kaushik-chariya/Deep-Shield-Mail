import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logger import logger
from src.utils.exception import MyException
from constants import MODEL_EVALUATION_MODEL_NAME, MODEL_PUSHER_ALIAS

import warnings
warnings.filterwarnings("ignore")


class ModelPusher:

    def __init__(self):
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = "kaushik-chariya"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        import dagshub
        dagshub.init(
            repo_owner = "kaushik-chariya",
            repo_name  = "Deep-Shield-Mail",
            mlflow     = True
        )

    def initiate_model_pusher(self, model_info: dict) -> None:
        try:
            logger.info("🚀 Model Pusher: STARTED")

            if not model_info.get('push_model', True):
                logger.info("⏭️ Model better nahi hai — Push SKIPPED")
                logger.info("👑 Model Pusher: COMPLETED (skipped)")
                return

            run_id     = model_info['run_id']
            model_name = MODEL_EVALUATION_MODEL_NAME  # ✅ Constants se
            source_uri = f"runs:/{run_id}/model"

            logger.info(f"📦 Model URI: {source_uri}")

            client = MlflowClient()

            try:
                client.create_registered_model(model_name)
                logger.info(f"📋 Registered model '{model_name}' created")
            except Exception:
                logger.info(f"📋 Registered model '{model_name}' already exists")

            model_version = client.create_model_version(
                name   = model_name,
                source = source_uri,
                run_id = run_id
            )
            logger.info(f"📝 Registered version: {model_version.version}")

            client.set_registered_model_alias(
                name    = model_name,
                alias   = MODEL_PUSHER_ALIAS,  # ✅ Constants se
                version = model_version.version
            )
            logger.info(f"✅ '{model_name}' v{model_version.version} → alias='{MODEL_PUSHER_ALIAS}' 🎯")

            logger.info("👑 Model Pusher: COMPLETED")

        except Exception as e:
            raise MyException(e, sys)


if __name__ == '__main__':
    import json
    with open('reports/experiment_info.json', 'r') as f:
        model_info = json.load(f)

    model_pusher = ModelPusher()
    model_pusher.initiate_model_pusher(model_info=model_info)