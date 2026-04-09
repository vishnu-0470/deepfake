"""
backend/routers/kyc.py
─────────────────────────────────────────────────────────
DeepShield KYC  –  REST Endpoints
  POST /api/kyc/session        create KYC session
  POST /api/kyc/upload-doc     upload ID document
  POST /api/kyc/upload-audio   receive audio chunk
  GET  /api/kyc/result/{id}    fetch analysis result
─────────────────────────────────────────────────────────
"""

import uuid
import secrets
import random
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from loguru import logger

from backend.config import get_settings, Settings
from backend.models.schemas import (
    KYCSessionRequest, KYCSessionResponse, KYCAnalysisResult
)
from backend.pipeline.session_store import session_store

router = APIRouter()

# Illumination challenge colors (high contrast)
ILLUM_COLORS = ["#FF0000", "#0000FF", "#00FF00", "#FFFFFF", "#FF00FF", "#00FFFF", "#FFFF00"]

# Liveness challenges
LIVENESS_CHALLENGES = [
    "Look LEFT",
    "Look RIGHT",
    "Blink TWICE",
    "Nod your head",
    "Smile",
]


def get_settings_dep() -> Settings:
    return get_settings()


# ── Session ──────────────────────────────────────────────────────────────────
@router.post("/session", response_model=KYCSessionResponse)
async def create_session(
    payload: KYCSessionRequest,
    settings: Annotated[Settings, Depends(get_settings_dep)],
):
    session_id    = str(uuid.uuid4())
    session_token = secrets.token_urlsafe(32)

    # Pick random illumination colors
    colors = random.sample(ILLUM_COLORS, settings.ILLUM_FLASH_COUNT)
    challenge = random.choice(LIVENESS_CHALLENGES)

    session_data = {
        "session_id":    session_id,
        "session_token": session_token,
        "applicant_name": payload.applicant_name,
        "id_type":       payload.id_type,
        "illum_colors":  colors,
        "liveness":      challenge,
        "status":        "ACTIVE",
        "doc_path":      None,
        "audio_paths":   [],
        "frames":        [],       # will hold raw frame bytes
        "result":        None,
    }
    await session_store.set(session_id, session_data)

    logger.info(f"[SESSION] Created {session_id} for {payload.applicant_name}")

    return KYCSessionResponse(
        session_id=session_id,
        session_token=session_token,
        illum_challenge_colors=colors,
        liveness_challenge=challenge,
    )


# ── Document Upload ──────────────────────────────────────────────────────────
@router.post("/upload-doc")
async def upload_document(
    file: Annotated[UploadFile, File(...)],
    session_id: Annotated[str, Form(...)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
):
    session = await session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Validate file type
    allowed = {".jpg", ".jpeg", ".png", ".pdf"}
    suffix  = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"File type {suffix} not allowed")

    # Save document
    save_path = settings.UPLOAD_DIR / f"{session_id}_doc{suffix}"
    contents  = await file.read()
    save_path.write_bytes(contents)

    session["doc_path"] = str(save_path)
    await session_store.set(session_id, session)

    logger.info(f"[DOC] Saved document for session {session_id} → {save_path}")
    return JSONResponse({"ok": True, "path": str(save_path)})


# ── Audio Chunk Upload ────────────────────────────────────────────────────────
@router.post("/upload-audio")
async def upload_audio(
    audio: Annotated[UploadFile, File(...)],
    session_id: Annotated[str, Form(...)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
):
    session = await session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    idx       = len(session.get("audio_paths", []))
    save_path = settings.UPLOAD_DIR / f"{session_id}_audio_{idx:03d}.webm"
    contents  = await audio.read()
    save_path.write_bytes(contents)

    session.setdefault("audio_paths", []).append(str(save_path))
    await session_store.set(session_id, session)

    return JSONResponse({"ok": True})


# ── Result Fetch ─────────────────────────────────────────────────────────────
@router.get("/result/{session_id}", response_model=KYCAnalysisResult)
async def get_result(session_id: str):
    session = await session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.get("result"):
        raise HTTPException(status_code=202, detail="Analysis still in progress")
    return session["result"]
