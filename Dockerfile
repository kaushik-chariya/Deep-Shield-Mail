FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install flask gunicorn scikit-learn numpy pandas dill from-root

# Copy required folders
COPY serving ./serving
COPY src ./src
COPY artifact ./artifact
COPY templates ./templates
COPY constants ./constants
COPY .project-root ./
# Python path
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "serving.api.app:app"]