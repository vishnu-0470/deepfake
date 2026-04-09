# ─────────────────────────────────────────────────────────
#  DeepShield KYC  –  Dockerfile
#  Multi-stage build: builder → runtime
# ─────────────────────────────────────────────────────────

# ── Stage 1: Builder (compile C binary + install Python deps) ──────────────
FROM python:3.11-slim AS builder

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    libgl1-mesa-glx libglib2.0-0 \
    ffmpeg \
    tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Compile C camera auth binary
COPY hardware/camera_auth.c ./hardware/
RUN gcc -O2 -o ./hardware/camera_auth ./hardware/camera_auth.c

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="DeepShield KYC Team"
LABEL description="AI-powered deepfake detection for Video KYC"

# Runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    ffmpeg \
    tesseract-ocr tesseract-ocr-eng \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 deepshield

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy compiled binary
COPY --from=builder /build/hardware/camera_auth ./hardware/camera_auth
RUN chmod +x ./hardware/camera_auth

# Copy application code
COPY backend/   ./backend/
COPY frontend/  ./frontend/
COPY .env.example .env

# Create required directories
RUN mkdir -p uploads models logs && \
    chown -R deepshield:deepshield /app

USER deepshield

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
