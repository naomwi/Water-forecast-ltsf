# =============================================================================
# HydroPred - Vertex AI Custom Training Container
# =============================================================================
# This Dockerfile packages the SpikeDLinear training pipeline for execution
# on Google Vertex AI Custom Training Jobs.
#
# Build & Push:
#   docker build -t gcr.io/YOUR_PROJECT_ID/hydropred-trainer:latest .
#   docker push gcr.io/YOUR_PROJECT_ID/hydropred-trainer:latest
# =============================================================================

FROM python:3.11-slim

# System dependencies for PyEMD / scipy / numpy compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (Docker layer caching optimization)
COPY requirements_vertex.txt /app/requirements_vertex.txt
RUN pip install --no-cache-dir --timeout 300 -r requirements_vertex.txt

# Speed up CEEMDAN for cloud execution
ENV CEEMDAN_TRIALS=10

# Copy only the necessary source code (not the entire project)
COPY Proposed_Models/src/ /app/Proposed_Models/src/
COPY scripts/train_vertex.py /app/train_vertex.py

# Default entrypoint: run the training script
# All arguments are passed at runtime by Vertex AI SDK
ENTRYPOINT ["python", "/app/train_vertex.py"]
