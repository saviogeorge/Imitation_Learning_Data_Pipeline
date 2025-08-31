# Dockerfile (root)
FROM python:3.8-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps (zstd for parquet compression, compilers for wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libzstd1 ca-certificates curl git build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Leverage layer caching: copy reqs first
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy code
COPY src ./src
COPY scripts ./scripts

# Default entrypoint: our CLI
ENTRYPOINT ["python", "scripts/neura.py"]
