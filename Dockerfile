# syntax=docker/dockerfile:1
# ──────────────────────────────────────────────────────────────────────────────
# Multi-stage Dockerfile — compatible with Docker and Podman (rootless)
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System build deps (for Pillow + cryptography wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
        libssl-dev \
        libjpeg-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Blockchain Medical FL"
LABEL org.opencontainers.image.description="Decentralised Federated Learning for Medical AI"

# Minimal runtime OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Create required directories
RUN mkdir -p instance data \
             app/static/uploads/lab_reports \
             app/static/uploads/verified_files \
             app/static/uploads/temp

# Non-root user (Podman rootless best-practice)
RUN useradd --uid 1001 --no-create-home --shell /bin/false appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

ENV FLASK_APP=run.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/').read()" || exit 1

# Use Gunicorn as the production WSGI server (4 workers, pre-fork model)
# Falls back to python3 run.py only when FLASK_ENV=development
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:5000", \
     "--timeout=120", "--access-logfile=-", "--error-logfile=-", \
     "run:app"]
