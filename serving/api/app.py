# ═══════════════════════════════════════════════════════════════
# app.py  —  Deep-Shield-Mail  |  Flask Serving App
# ═══════════════════════════════════════════════════════════════

import os
import sys
from flask import Flask, request, jsonify, render_template

from src.utils.logger    import logger
from src.utils.exception import MyException

# ── Lazy-load prediction pipeline (loaded once on first request) ──
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.pipeline.prediction_pipeline import PredictionPipeline
        logger.info("🔄 Loading PredictionPipeline …")
        _pipeline = PredictionPipeline()
        logger.info("✅ PredictionPipeline ready")
    return _pipeline


# ── Flask app ─────────────────────────────────────────────────────
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "templates"), static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "static"))


# ═══════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    """Serve the main UI."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST  /predict
    Body (JSON):  { "email": "<raw email text>" }

    Response:
    {
        "label"      : "Spam" | "Not Spam",
        "prediction" : 1      | 0,
        "probability": float
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        email_text = data.get("email", "").strip()

        if not email_text:
            return jsonify({"error": "Email text is required."}), 400

        pipeline = get_pipeline()
        result   = pipeline.predict(email_text)

        logger.info("📨 /predict → %s  (prob=%.4f)", result["label"], result["probability"])
        return jsonify(result), 200

    except MyException as e:
        logger.error("MyException in /predict: %s", str(e))
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.error("Unexpected error in /predict: %s", str(e))
        return jsonify({"error": "Internal server error."}), 500


@app.route("/train", methods=["GET", "POST"])
def train():
    """
    Trigger the full training pipeline.
    GET  /train  →  starts training and returns status.
    """
    try:
        from src.pipeline.training_pipeline import TrainingPipeline

        logger.info("🏋️  Training pipeline triggered via /train endpoint")
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()

        # Reset cached prediction pipeline so it picks up new model
        global _pipeline
        _pipeline = None

        logger.info("✅ Training complete")
        return jsonify({"status": "success", "message": "Training completed successfully."}), 200

    except MyException as e:
        logger.error("MyException in /train: %s", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

    except Exception as e:
        logger.error("Unexpected error in /train: %s", str(e))
        return jsonify({"status": "error", "message": "Training failed. Check logs."}), 500


@app.route("/health", methods=["GET"])
def health():
    """Basic health check endpoint."""
    return jsonify({"status": "ok", "service": "Deep-Shield-Mail"}), 200


# ═══════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    PORT = int(os.getenv("FLASK_PORT", 8000))
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    logger.info("🚀 Starting Deep-Shield-Mail on %s:%s (debug=%s)", HOST, PORT, DEBUG)
    app.run(host=HOST, port=PORT, debug=DEBUG)