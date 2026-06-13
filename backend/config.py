"""
backend/config.py
─────────────────────────────────────────────────────────
DeepShield KYC  –  Centralised Configuration
All secrets come from environment variables / .env file.
─────────────────────────────────────────────────────────
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────
    APP_NAME: str = "DeepShield KYC"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    SECRET_KEY: str = Field(..., env="SECRET_KEY")

    # ── Server ───────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # ── Redis (session & task queue) ─────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Storage ──────────────────────────────────────────
    UPLOAD_DIR: Path = ROOT_DIR / "uploads"
    MODEL_DIR: Path = ROOT_DIR / "models"
    LOG_DIR: Path = ROOT_DIR / "logs"

    # ── Detection Thresholds ─────────────────────────────
    # Risk score above which KYC is BLOCKED
    BLOCK_THRESHOLD: int = 70
    # Risk score above which KYC is sent for MANUAL REVIEW
    REVIEW_THRESHOLD: int = 40

    # Layer weights (must sum to 1.0)
    WEIGHT_DEEPFAKE_CLASSIFIER: float = 0.30
    WEIGHT_RPPG: float = 0.25
    WEIGHT_ACOUSTIC: float = 0.20
    WEIGHT_ILLUMINATION: float = 0.15
    WEIGHT_FACE_MATCH: float = 0.10

    # ── rPPG ─────────────────────────────────────────────
    RPPG_MIN_HZ: float = 0.75     # 45 bpm
    RPPG_MAX_HZ: float = 4.0      # 240 bpm
    RPPG_WINDOW_SEC: int = 10     # seconds of video for signal
    RPPG_FPS: int = 30

    # ── Acoustic ─────────────────────────────────────────
    ACOUSTIC_SAMPLE_RATE: int = 16000
    ACOUSTIC_MIN_SNR_DB: float = 5.0   # below this → too clean → suspicious
    ACOUSTIC_REVERB_MIN_RT60: float = 0.05  # seconds

    # ── Illumination Challenge ────────────────────────────
    ILLUM_FLASH_COUNT: int = 4
    ILLUM_FLASH_DURATION_MS: int = 300
    ILLUM_CORRELATION_THRESHOLD: float = 0.65

    # ── Face Match ───────────────────────────────────────
    FACE_MATCH_MIN_COSINE: float = 0.40

    # ── Deepfake Classifier ──────────────────────────────
    CLASSIFIER_MODEL_NAME: str = "efficientnet_b4"
    CLASSIFIER_CHECKPOINT: str = "deepfake_effnetb4_ff++.pth"
    CLASSIFIER_FAKE_THRESHOLD: float = 0.55   # probability above = FAKE

    # ── OCR ──────────────────────────────────────────────
    TESSERACT_CMD: str = "/usr/bin/tesseract"

    # ── Hardware Auth ────────────────────────────────────
    HW_AUTH_BINARY: Path = ROOT_DIR / "hardware" / "camera_auth"
    HW_AUTH_ENABLED: bool = True

    # ── CORS ─────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8080"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
