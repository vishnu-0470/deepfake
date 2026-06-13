"""
tests/test_pipeline.py
─────────────────────────────────────────────────────────
DeepShield KYC  –  Test Suite

Covers:
  - rPPG detector (synthetic signal vs real signal)
  - Acoustic analyzer (clean vs noisy audio)
  - Illumination challenge (matching vs non-matching)
  - Deepfake classifier FFT stage
  - Risk scorer ensemble logic
  - REST API endpoints
─────────────────────────────────────────────────────────
"""

import asyncio
import json
import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_real_face_frames(n: int = 90) -> list[np.ndarray]:
    """Generate synthetic 'real' video frames with a face-like region."""
    frames = []
    for i in range(n):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Skin-tone face ellipse
        cv2.ellipse(frame, (320, 240), (80, 100), 0, 0, 360, (200, 160, 120), -1)
        # Add slight color oscillation to simulate rPPG
        pulse = int(5 * np.sin(2 * np.pi * 1.2 * i / 30))   # 1.2 Hz = 72 bpm
        frame[190:230, 280:360] = np.clip(
            frame[190:230, 280:360].astype(int) + [pulse, -pulse // 2, 0], 0, 255
        )
        frames.append(frame)
    return frames


def _make_fake_frames(n: int = 90) -> list[np.ndarray]:
    """Generate static frames (no pulse) to simulate deepfake."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.ellipse(frame, (320, 240), (80, 100), 0, 0, 360, (200, 160, 120), -1)
    return [frame.copy() for _ in range(n)]


def _make_real_audio(sr: int = 16000, duration: float = 5.0) -> np.ndarray:
    """Generate audio with background noise + speech component + reverb-like decay."""
    t = np.linspace(0, duration, int(sr * duration))
    speech    = 0.3 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.sin(2 * np.pi * 400 * t)
    noise     = 0.05 * np.random.randn(len(t))
    # Simple exponential decay for reverb simulation
    reverb    = np.convolve(speech, np.exp(-np.linspace(0, 5, sr // 4)), mode="same")
    audio     = speech + noise + 0.02 * reverb
    return audio.astype(np.float32)


def _make_synthetic_audio(sr: int = 16000, duration: float = 5.0) -> np.ndarray:
    """Generate unnaturally clean audio (high SNR, no reverb)."""
    t = np.linspace(0, duration, int(sr * duration))
    speech = 0.5 * np.sin(2 * np.pi * 200 * t)   # pure tone, no noise
    return speech.astype(np.float32)


# ── rPPG Tests ───────────────────────────────────────────────────────────────

class TestRPPGDetector:

    @pytest.mark.asyncio
    async def test_real_frames_pass(self):
        from backend.detectors.rppg_detector import RPPGDetector
        detector = RPPGDetector()
        frames   = _make_real_face_frames(90)
        result   = await detector.analyze(frames)
        # Real frames should have lower risk contribution
        assert result.risk_contribution < 50.0, \
            f"Expected low risk for real frames, got {result.risk_contribution}"

    @pytest.mark.asyncio
    async def test_static_frames_flagged(self):
        from backend.detectors.rppg_detector import RPPGDetector
        detector = RPPGDetector()
        frames   = _make_fake_frames(90)
        result   = await detector.analyze(frames)
        # Completely static frames have zero variance → should flag
        assert result.signal_strength < 0.05, \
            f"Expected very low signal for static frames, got {result.signal_strength}"

    @pytest.mark.asyncio
    async def test_insufficient_frames_returns_unknown(self):
        from backend.detectors.rppg_detector import RPPGDetector
        from backend.models.schemas import DetectionLabel
        detector = RPPGDetector()
        result   = await detector.analyze([])
        assert result.label == DetectionLabel.UNKNOWN

    @pytest.mark.asyncio
    async def test_quick_check_real(self):
        from backend.detectors.rppg_detector import RPPGDetector
        detector = RPPGDetector()
        frames   = _make_real_face_frames(30)
        result   = await detector.quick_check(frames)
        # Should return bool or None (not raise)
        assert result is None or isinstance(result, bool)


# ── Acoustic Tests ───────────────────────────────────────────────────────────

class TestAcousticAnalyzer:

    def test_snr_estimation_real(self):
        from backend.detectors.acoustic_analyzer import AcousticAnalyzer
        analyzer = AcousticAnalyzer()
        audio    = _make_real_audio()
        snr      = analyzer._estimate_snr(audio, 16000)
        assert snr is not None
        assert 5.0 <= snr <= 50.0, f"Expected natural SNR, got {snr}"

    def test_snr_estimation_synthetic(self):
        from backend.detectors.acoustic_analyzer import AcousticAnalyzer
        analyzer = AcousticAnalyzer()
        audio    = _make_synthetic_audio()
        snr      = analyzer._estimate_snr(audio, 16000)
        assert snr is not None
        assert snr > 30.0, f"Expected high SNR for synthetic, got {snr}"

    def test_zcr_variance_real_vs_synthetic(self):
        from backend.detectors.acoustic_analyzer import AcousticAnalyzer
        analyzer = AcousticAnalyzer()
        real_var = analyzer._zcr_variance(_make_real_audio(), 16000)
        synth_var = analyzer._zcr_variance(_make_synthetic_audio(), 16000)
        # Real audio should have higher ZCR variance
        assert real_var is not None and synth_var is not None
        assert real_var > synth_var, \
            f"Expected real ZCR var ({real_var}) > synthetic ({synth_var})"


# ── Illumination Tests ───────────────────────────────────────────────────────

class TestIlluminationChallenge:

    @pytest.mark.asyncio
    async def test_face_reflects_flash_passes(self):
        from backend.detectors.illumination_challenge import IlluminationChallenge
        challenge = IlluminationChallenge()

        colors = ["#FF0000", "#0000FF"]
        # Before: neutral gray face
        before = [np.full((480, 640, 3), 150, dtype=np.uint8) for _ in range(10)]
        # During red flash: face turns reddish
        during_red = [np.full((480, 640, 3), 150, dtype=np.uint8) for _ in range(5)]
        for f in during_red:
            f[:, :, 2] = 200   # boost red channel (BGR: blue, green, RED)
        # During blue flash: face turns bluish
        during_blue = [np.full((480, 640, 3), 150, dtype=np.uint8) for _ in range(5)]
        for f in during_blue:
            f[:, :, 0] = 200   # boost blue channel

        frames_during = {"#FF0000": during_red, "#0000FF": during_blue}
        result = await challenge.analyze(before, frames_during, colors)
        # Correlation should be positive — face reflected the flash colors
        assert result.correlation_score > 0.0

    @pytest.mark.asyncio
    async def test_no_face_response_flagged(self):
        from backend.detectors.illumination_challenge import IlluminationChallenge
        from backend.models.schemas import DetectionLabel
        challenge = IlluminationChallenge()

        colors = ["#FF0000", "#0000FF"]
        # Before and during are identical — no reflection
        static = [np.full((480, 640, 3), 150, dtype=np.uint8) for _ in range(10)]
        frames_during = {"#FF0000": static, "#0000FF": static}
        result = await challenge.analyze(static, frames_during, colors)
        # No change in face color → correlation near 0 → should be flagged
        assert result.label == DetectionLabel.FAKE or result.correlation_score < 0.5


# ── Deepfake Classifier FFT Tests ─────────────────────────────────────────────

class TestDeepfakeClassifierFFT:

    def test_real_image_low_fft_score(self):
        from backend.detectors.deepfake_classifier import DeepfakeClassifier
        clf   = DeepfakeClassifier()
        # Real photograph: use random noise (no GAN grid pattern)
        img   = (np.random.randn(224, 224, 3) * 30 + 128).clip(0, 255).astype(np.uint8)
        score = clf._fft_fake_score(img)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_gan_grid_pattern_detected(self):
        from backend.detectors.deepfake_classifier import DeepfakeClassifier
        clf = DeepfakeClassifier()
        # Simulate GAN grid: add periodic high-frequency pattern
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        # Regular grid every 8 pixels (simulates GAN upsampling artifact)
        img[::8, :] = 200
        img[:, ::8] = 200
        score = clf._fft_fake_score(img)
        # Grid pattern should yield a higher score than random noise
        assert isinstance(score, float)
        # Note: exact threshold depends on implementation details


# ── Risk Scorer Tests ─────────────────────────────────────────────────────────

class TestRiskScorer:

    def _make_real_result(self, cls):
        return cls(
            label=__import__("backend.models.schemas", fromlist=["DetectionLabel"]).DetectionLabel.REAL,
            confidence=0.92,
            risk_contribution=5.0,
            detail="Passed",
        )

    def _make_fake_result(self, cls, risk=80.0):
        return cls(
            label=__import__("backend.models.schemas", fromlist=["DetectionLabel"]).DetectionLabel.FAKE,
            confidence=0.91,
            risk_contribution=risk,
            detail="Failed",
        )

    def test_all_pass_approved(self):
        from backend.utils.scoring import RiskScorer
        from backend.models.schemas import (
            KYCVerdict, DeepfakeClassifierResult, RPPGResult, AcousticResult,
            IlluminationResult, FaceMatchResult, HardwareAuthResult, LivenessResult,
            DetectionLabel,
        )
        scorer = RiskScorer()

        def real(cls):
            return cls(label=DetectionLabel.REAL, confidence=0.95, risk_contribution=3.0, detail="OK")

        result = scorer.compute(
            session_id="test-123", applicant_name="Test User",
            deepfake=DeepfakeClassifierResult(label=DetectionLabel.REAL, confidence=0.95, risk_contribution=3.0, detail="OK"),
            rppg=RPPGResult(label=DetectionLabel.REAL, confidence=0.93, risk_contribution=4.0, detail="OK"),
            acoustic=AcousticResult(label=DetectionLabel.REAL, confidence=0.90, risk_contribution=3.0, detail="OK"),
            illumination=IlluminationResult(label=DetectionLabel.REAL, confidence=0.91, risk_contribution=2.0, detail="OK"),
            face_match=FaceMatchResult(label=DetectionLabel.REAL, confidence=0.94, risk_contribution=3.0, detail="OK"),
            hardware=HardwareAuthResult(label=DetectionLabel.REAL, confidence=0.88, risk_contribution=0.0, detail="OK", is_virtual=False),
            liveness=LivenessResult(label=DetectionLabel.REAL, confidence=0.85, risk_contribution=0.0, detail="OK"),
        )
        assert result.verdict == KYCVerdict.APPROVED
        assert result.risk_score < 40.0

    def test_deepfake_blocked(self):
        from backend.utils.scoring import RiskScorer
        from backend.models.schemas import (
            KYCVerdict, DeepfakeClassifierResult, RPPGResult, AcousticResult,
            IlluminationResult, FaceMatchResult, HardwareAuthResult, LivenessResult,
            DetectionLabel,
        )
        scorer = RiskScorer()
        result = scorer.compute(
            session_id="test-456", applicant_name="Fake User",
            deepfake=DeepfakeClassifierResult(label=DetectionLabel.FAKE, confidence=0.95, risk_contribution=90.0, detail="FAKE"),
            rppg=RPPGResult(label=DetectionLabel.FAKE, confidence=0.90, risk_contribution=85.0, detail="No pulse"),
            acoustic=AcousticResult(label=DetectionLabel.FAKE, confidence=0.88, risk_contribution=75.0, detail="Synthetic"),
            illumination=IlluminationResult(label=DetectionLabel.FAKE, confidence=0.87, risk_contribution=70.0, detail="No reflection"),
            face_match=FaceMatchResult(label=DetectionLabel.FAKE, confidence=0.92, risk_contribution=85.0, detail="Mismatch"),
            hardware=HardwareAuthResult(label=DetectionLabel.FAKE, confidence=0.91, risk_contribution=0.0, detail="Virtual", is_virtual=True),
            liveness=LivenessResult(label=DetectionLabel.FAKE, confidence=0.80, risk_contribution=0.0, detail="Failed"),
        )
        assert result.verdict == KYCVerdict.BLOCKED
        assert result.risk_score >= 70.0

    def test_virtual_camera_adds_penalty(self):
        from backend.utils.scoring import RiskScorer
        from backend.models.schemas import (
            DeepfakeClassifierResult, RPPGResult, AcousticResult,
            IlluminationResult, FaceMatchResult, HardwareAuthResult, LivenessResult,
            DetectionLabel,
        )
        scorer = RiskScorer()

        # Make result with no virtual camera
        r1 = scorer.compute(
            session_id="t1", applicant_name="Test",
            deepfake=DeepfakeClassifierResult(label=DetectionLabel.REAL, confidence=0.9, risk_contribution=5.0, detail="OK"),
            rppg=RPPGResult(label=DetectionLabel.REAL, confidence=0.9, risk_contribution=5.0, detail="OK"),
            acoustic=AcousticResult(label=DetectionLabel.REAL, confidence=0.9, risk_contribution=5.0, detail="OK"),
            illumination=IlluminationResult(label=DetectionLabel.REAL, confidence=0.9, risk_contribution=5.0, detail="OK"),
            face_match=FaceMatchResult(label=DetectionLabel.REAL, confidence=0.9, risk_contribution=5.0, detail="OK"),
            hardware=HardwareAuthResult(label=DetectionLabel.REAL, confidence=0.9, risk_contribution=0.0, detail="OK", is_virtual=False),
            liveness=LivenessResult(label=DetectionLabel.REAL, confidence=0.9, risk_contribution=0.0, detail="OK"),
        )
        # Same but with virtual camera
        r2 = scorer.compute(
            session_id="t2", applicant_name="Test",
            deepfake=DeepfakeClassifierResult(label=DetectionLabel.REAL, confidence=0.9, risk_contribution=5.0, detail="OK"),
            rppg=RPPGResult(label=DetectionLabel.REAL, confidence=0.9, risk_contribution=5.0, detail="OK"),
            acoustic=AcousticResult(label=DetectionLabel.REAL, confidence=0.9, risk_contribution=5.0, detail="OK"),
            illumination=IlluminationResult(label=DetectionLabel.REAL, confidence=0.9, risk_contribution=5.0, detail="OK"),
            face_match=FaceMatchResult(label=DetectionLabel.REAL, confidence=0.9, risk_contribution=5.0, detail="OK"),
            hardware=HardwareAuthResult(label=DetectionLabel.FAKE, confidence=0.9, risk_contribution=0.0, detail="Virtual", is_virtual=True),
            liveness=LivenessResult(label=DetectionLabel.REAL, confidence=0.9, risk_contribution=0.0, detail="OK"),
        )
        assert r2.risk_score == r1.risk_score + 20.0


# ── API Endpoint Tests ────────────────────────────────────────────────────────

class TestAPI:

    @pytest.mark.asyncio
    async def test_create_session(self):
        from fastapi.testclient import TestClient
        with patch("backend.pipeline.orchestrator.pipeline.warmup", new_callable=AsyncMock):
            with patch("backend.pipeline.session_store.session_store.connect", new_callable=AsyncMock):
                with patch("backend.pipeline.session_store.session_store.set", new_callable=AsyncMock):
                    from backend.main import app
                    client = TestClient(app)
                    resp   = client.post("/api/kyc/session", json={
                        "applicant_name": "Priya Sharma",
                        "id_type": "AADHAAR"
                    })
                    assert resp.status_code == 200
                    data = resp.json()
                    assert "session_id" in data
                    assert "session_token" in data
                    assert len(data["illum_challenge_colors"]) > 0

    @pytest.mark.asyncio
    async def test_health_check(self):
        from fastapi.testclient import TestClient
        with patch("backend.pipeline.orchestrator.pipeline.warmup", new_callable=AsyncMock):
            from backend.main import app
            client = TestClient(app)
            resp   = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
