import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.logger import logger

# ── Flask App ─────────────────────────────────────────────────
app = Flask(__name__, template_folder="../../templates", static_folder="../../static")

# ── Prometheus Metrics ────────────────────────────────────────
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total number of requests to the app",
    ["method", "endpoint"],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Latency of requests in seconds",
    ["endpoint"],
    registry=registry
)

PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Count of predictions for each class",
    ["prediction"],
    registry=registry
)

# ── Load Pipeline once at startup ─────────────────────────────
logger.info("Loading PredictionPipeline...")
pipeline = PredictionPipeline()
logger.info("✅ PredictionPipeline loaded successfully!")


# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()

    response = render_template("index.html", result=None)

    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response


@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    try:
        text = request.form.get("text", "").strip()

        if not text:
            return render_template("index.html", result="error", message="Email text cannot be empty!")

        # ── Predict ───────────────────────────────────────────
        result = pipeline.predict(text)

        # ── Track prediction in Prometheus ────────────────────
        PREDICTION_COUNT.labels(prediction=result["label"]).inc()

        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

        return render_template(
            "index.html",
            result=result["label"],            # "Spam" or "Not Spam"
            probability=result["probability"],  # 0.9906
            prediction=result["prediction"],    # 0 or 1
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return render_template("index.html", result="error", message=str(e))


@app.route("/predict_api", methods=["POST"])
def predict_api():
    """JSON API endpoint — Postman/curl ke liye"""
    REQUEST_COUNT.labels(method="POST", endpoint="/predict_api").inc()
    start_time = time.time()

    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Email text cannot be empty!"}), 400

        result = pipeline.predict(text)

        PREDICTION_COUNT.labels(prediction=result["label"]).inc()
        REQUEST_LATENCY.labels(endpoint="/predict_api").observe(time.time() - start_time)

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"API Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

 
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)