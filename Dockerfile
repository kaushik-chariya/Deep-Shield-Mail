FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install \
    flask \
    gunicorn \
    scikit-learn \
    numpy \
    pandas \
    dill \
    from-root \
    google-auth-oauthlib \
    google-api-python-client

COPY serving ./serving
COPY src ./src
COPY artifact ./artifact
COPY templates ./templates
COPY static ./static
COPY constants ./constants
COPY config ./config


# Copy root files
COPY params.yaml .
COPY setup.py .
COPY pyproject.toml .
COPY .project-root ./
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["gunicorn", "--workers", "2", "--threads", "2", "--bind", "0.0.0.0:8000", "serving.api.app:app"]