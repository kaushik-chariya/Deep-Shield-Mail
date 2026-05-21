FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY setup.py .
COPY pyproject.toml .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install -e . --no-deps

COPY serving ./serving
COPY src ./src
COPY templates ./templates
COPY static ./static
COPY constants ./constants
COPY config ./config
COPY params.yaml .
COPY .project-root ./

# Copy only the latest model artifacts — not all timestamped runs
COPY artifact/data_transformation/transformed_object ./artifact/data_transformation/transformed_object
COPY artifact/model_trainer/trained_model ./artifact/model_trainer/trained_model

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["gunicorn", "--workers", "2", "--threads", "2", "--bind", "0.0.0.0:8000", "serving.api.app:app"]