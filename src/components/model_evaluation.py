import numpy as np
import glob
import pickle
import json
import os
import sys

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src.utils.logger import logger
from src.utils.exception import MyException
from constants import MODEL_EVALUATION_EXPERIMENT_NAME, MODEL_EVALUATION_MODEL_NAME

import warnings
warnings.filterwarnings("ignore")


class ModelEvaluation:

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

    def _get_latest_model_path(self, base_dir: str = './artifact') -> str:
        model_files = glob.glob(f'{base_dir}/**/model.pkl', recursive=True)
        if not model_files:
            raise FileNotFoundError(f"No model.pkl found inside '{base_dir}'")
        latest = max(model_files, key=os.path.getmtime)
        logger.info(f'Latest model found at: {latest}')
        return latest

    def _get_latest_test_data_path(self, base_dir: str = './artifact') -> str:
        test_files = glob.glob(f'{base_dir}/**/test.npy', recursive=True)
        if not test_files:
            raise FileNotFoundError(f"No test.npy found inside '{base_dir}'")
        latest = max(test_files, key=os.path.getmtime)
        logger.info(f'Latest test data found at: {latest}')
        return latest

    def _load_model(self, file_path: str):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f'Model loaded from {file_path}')
        return model

    def _load_test_data(self, file_path: str) -> np.ndarray:
        data = np.load(file_path)
        logger.info(f'Test data loaded | Shape: {data.shape}')
        return data

    def _evaluate(self, clf, X_test, y_test) -> dict:
        y_pred       = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        return {
            'accuracy' : accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall'   : recall_score(y_test, y_pred),
            'auc'      : roc_auc_score(y_test, y_pred_proba)
        }

    def _get_best_staging_accuracy(self, model_name: str) -> float:
        try:
            client = MlflowClient()
            staging_version = client.get_model_version_by_alias(model_name, "staging")
            prev_run = client.get_run(staging_version.run_id)
            best_acc = prev_run.data.metrics.get("accuracy", 0.0)
            logger.info(f"📊 Previous staging accuracy: {best_acc:.4f}")
            return best_acc
        except Exception:
            logger.info("📊 No staging model found — pehli baar push hoga")
            return 0.0

    def initiate_model_evaluation(self) -> dict:
        try:
            logger.info("=" * 60)
            logger.info("Model Evaluation: STARTED")
            logger.info("=" * 60)

            # ✅ Constants se lo — hardcoded nahi
            model_name = MODEL_EVALUATION_MODEL_NAME
            mlflow.set_experiment(MODEL_EVALUATION_EXPERIMENT_NAME)

            with mlflow.start_run() as run:

                model_path = self._get_latest_model_path('./artifact')
                clf        = self._load_model(model_path)

                test_data_path = self._get_latest_test_data_path('./artifact')
                test_arr       = self._load_test_data(test_data_path)
                X_test = test_arr[:, :-1]
                y_test = test_arr[:, -1]

                logger.info(f"X_test: {X_test.shape} | y_test: {y_test.shape}")

                metrics = self._evaluate(clf, X_test, y_test)

                logger.info(f"Accuracy : {metrics['accuracy']:.4f}  🏆")
                logger.info(f"Precision: {metrics['precision']:.4f} 🏆")
                logger.info(f"Recall   : {metrics['recall']:.4f}   🏆")
                logger.info(f"AUC      : {metrics['auc']:.4f}      🏆")

                os.makedirs('reports', exist_ok=True)
                with open('reports/metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=4)
                logger.info("Metrics saved to reports/metrics.json")

                for k, v in metrics.items():
                    mlflow.log_metric(k, v)

                if hasattr(clf, 'get_params'):
                    for k, v in clf.get_params().items():
                        mlflow.log_param(k, v)

                mlflow.xgboost.log_model(
                    xgb_model     = clf,
                    name = "model"
                )
                logger.info("Model logged to MLflow under artifact_path='model'")

                mlflow.log_artifact('reports/metrics.json')

                best_accuracy = self._get_best_staging_accuracy(model_name)
                should_push   = metrics['accuracy'] > best_accuracy

                if should_push:
                    logger.info(f"✅ New model better hai ({metrics['accuracy']:.4f} > {best_accuracy:.4f}) — PUSH hoga 🚀")
                else:
                    logger.info(f"⏭️ New model better nahi ({metrics['accuracy']:.4f} <= {best_accuracy:.4f}) — SKIP")

                model_info = {
                    'run_id'    : run.info.run_id,
                    'model_path': 'model',
                    'push_model': should_push
                }
                with open('reports/experiment_info.json', 'w') as f:
                    json.dump(model_info, f, indent=4)

                logger.info("Model Evaluation: COMPLETED 👑")
                return model_info

        except Exception as e:
            raise MyException(e, sys)


if __name__ == '__main__':
    model_evaluation = ModelEvaluation()
    model_evaluation.initiate_model_evaluation()