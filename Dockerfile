# ─────────────────────────────────────────────────────────
#  DeepShield KYC  –  Dockerfile  (Render 512MB optimised)
# ─────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL maintainer="DeepShield KYC"

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Compile C camera auth binary
COPY hardware/camera_auth.c ./hardware/
RUN gcc -O2 -o ./hardware/camera_auth ./hardware/camera_auth.c && \
    chmod +x ./hardware/camera_auth

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/   ./backend/
COPY frontend/  ./frontend/

# Create required directories
RUN mkdir -p uploads models logs

EXPOSE 8000

# Single worker to stay within 512MB RAM
CMD ["python", "-m", "uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
