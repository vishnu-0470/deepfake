"""
backend/main.py
─────────────────────────────────────────────────────────
DeepShield KYC  –  FastAPI Application Entry Point
─────────────────────────────────────────────────────────
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from backend.config import get_settings
from backend.routers import kyc, ws_router
from backend.routers import otp
from backend.pipeline.orchestrator import pipeline  # singleton

settings = get_settings()


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown tasks."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Create required directories
    for d in [settings.UPLOAD_DIR, settings.MODEL_DIR, settings.LOG_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Warm up ML models (loads into GPU memory)
    await pipeline.warmup()

    logger.info("All models loaded. Server ready.")
    yield

    logger.info("Shutting down DeepShield KYC.")
    await pipeline.shutdown()


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Multi-layer AI-powered deepfake detection for Video KYC",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus metrics ────────────────────────────────────────────────────────
Instrumentator().instrument(app).expose(app)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(kyc.router,       prefix="/api/kyc",  tags=["KYC"])
app.include_router(otp.router,       prefix="/api/otp",  tags=["OTP"])
app.include_router(ws_router.router, prefix="/ws",       tags=["WebSocket"])

# ── Static files ──────────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


# ── Dev runner ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_level="debug" if settings.DEBUG else "info",
    )
