"""
backend/models/schemas.py
─────────────────────────────────────────────────────────
DeepShield KYC  –  Pydantic Data Models
─────────────────────────────────────────────────────────
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


# ── Enums ────────────────────────────────────────────────────────────────────

class KYCVerdict(str, Enum):
    APPROVED = "APPROVED"
    BLOCKED  = "BLOCKED"
    REVIEW   = "REVIEW"
    PENDING  = "PENDING"


class DetectionLabel(str, Enum):
    REAL    = "REAL"
    FAKE    = "FAKE"
    UNKNOWN = "UNKNOWN"


class FraudType(str, Enum):
    FACE_SWAP       = "FACE_SWAP"
    GAN_GENERATED   = "GAN_GENERATED"
    AUDIO_SYNTHETIC = "AUDIO_SYNTHETIC"
    VIRTUAL_CAMERA  = "VIRTUAL_CAMERA"
    ID_MISMATCH     = "ID_MISMATCH"
    LIVENESS_FAIL   = "LIVENESS_FAIL"
    NONE            = "NONE"


# ── Per-Layer Results ────────────────────────────────────────────────────────

class LayerResult(BaseModel):
    name: str
    label: DetectionLabel
    confidence: float = Field(ge=0.0, le=1.0)
    risk_contribution: float = Field(ge=0.0, le=100.0)
    detail: str = ""
    latency_ms: float = 0.0


class DeepfakeClassifierResult(LayerResult):
    name: str = "Deepfake Classifier"
    frames_analyzed: int = 0
    top_fake_frame_score: float = 0.0


class RPPGResult(LayerResult):
    name: str = "rPPG Biological Signal"
    estimated_bpm: Optional[float] = None
    dominant_frequency: Optional[float] = None
    signal_strength: float = 0.0


class AcousticResult(LayerResult):
    name: str = "Acoustic Profiling"
    estimated_snr_db: Optional[float] = None
    rt60_ms: Optional[float] = None
    has_reverb: bool = False


class IlluminationResult(LayerResult):
    name: str = "Illumination Challenge"
    challenges_sent: int = 0
    challenges_passed: int = 0
    correlation_score: float = 0.0


class FaceMatchResult(LayerResult):
    name: str = "Face Match (ID vs Selfie)"
    cosine_similarity: float = 0.0
    faces_detected_video: int = 0
    faces_detected_doc: int = 0


class HardwareAuthResult(LayerResult):
    name: str = "Hardware Camera Auth"
    device_name: Optional[str] = None
    is_virtual: bool = False
    vendor_id: Optional[str] = None


class LivenessResult(LayerResult):
    name: str = "Liveness Detection"
    challenge_type: str = ""
    user_response: str = ""


# ── KYC Session ──────────────────────────────────────────────────────────────

class KYCSessionRequest(BaseModel):
    applicant_name: str
    id_type: str = Field(..., description="AADHAAR | PAN | PASSPORT")
    session_token: Optional[str] = None


class KYCSessionResponse(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_token: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    illum_challenge_colors: list[str] = []
    liveness_challenge: str = ""


class KYCAnalysisResult(BaseModel):
    session_id: str
    applicant_name: str

    # Individual layer results
    deepfake_result: DeepfakeClassifierResult
    rppg_result: RPPGResult
    acoustic_result: AcousticResult
    illumination_result: IlluminationResult
    face_match_result: FaceMatchResult
    hardware_result: HardwareAuthResult
    liveness_result: LivenessResult

    # Ensemble
    risk_score: float = Field(ge=0.0, le=100.0)
    verdict: KYCVerdict
    fraud_types: list[FraudType] = []
    explanation: str = ""

    total_latency_ms: float = 0.0
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


# ── WebSocket Messages ───────────────────────────────────────────────────────

class WSMessageType(str, Enum):
    FRAME_ACK          = "FRAME_ACK"
    LAYER_UPDATE       = "LAYER_UPDATE"
    CHALLENGE_READY    = "CHALLENGE_READY"
    CHALLENGE_COMPLETE = "CHALLENGE_COMPLETE"
    ANALYSIS_COMPLETE  = "ANALYSIS_COMPLETE"
    ERROR              = "ERROR"


class WSMessage(BaseModel):
    type: WSMessageType
    payload: dict = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)
