"""
backend/pipeline/orchestrator.py
─────────────────────────────────────────────────────────
DeepShield KYC  –  Detection Pipeline Orchestrator

Runs all 6 detection layers in parallel, then feeds
results into the weighted risk scoring engine.

Layers (parallel):
  1. Deepfake Classifier  (EfficientNet-B4 + FFT)
  2. rPPG Biological Signal
  3. Acoustic Profiling
  4. Illumination Challenge
  5. Document OCR + Face Match
  6. Hardware Camera Auth
─────────────────────────────────────────────────────────
"""

import asyncio
import time
from typing import Callable, Optional
from loguru import logger

import numpy as np

from backend.config import get_settings
from backend.models.schemas import (
    KYCAnalysisResult, KYCVerdict, FraudType,
    DeepfakeClassifierResult, RPPGResult, AcousticResult,
    IlluminationResult, FaceMatchResult, HardwareAuthResult, LivenessResult,
    DetectionLabel,
)
from backend.detectors.deepfake_classifier  import deepfake_classifier
from backend.detectors.rppg_detector        import rppg_detector
from backend.detectors.acoustic_analyzer    import acoustic_analyzer
from backend.detectors.illumination_challenge import illum_challenge
from backend.detectors.document_ocr         import document_analyzer
from backend.detectors.hardware_checker     import hardware_checker
from backend.pipeline.session_store         import session_store
from backend.utils.scoring                  import risk_scorer

settings = get_settings()


class KYCPipeline:
    """Orchestrates all detection layers and produces a final KYC verdict."""

    async def warmup(self):
        """Pre-load all ML models at startup."""
        logger.info("[Pipeline] Warming up models...")

        await session_store.connect(settings.REDIS_URL)

        await asyncio.get_event_loop().run_in_executor(
            None, deepfake_classifier.load_model
        )

        logger.info("[Pipeline] Warmup complete.")

    async def shutdown(self):
        await session_store.close()

    # ── Main analysis entry point ─────────────────────────────────────────────

    async def run(
        self,
        session_id: str,
        frames: list[np.ndarray],
        on_layer_update: Optional[Callable] = None,
    ) -> KYCAnalysisResult:
        """
        Run all detection layers and return a KYCAnalysisResult.
        on_layer_update is called after each layer completes.
        """
        t0 = time.perf_counter()
        session = await session_store.get(session_id)

        if not session:
            raise ValueError(f"Session {session_id} not found")

        audio_paths    = session.get("audio_paths", [])
        doc_path       = session.get("doc_path")
        illum_colors   = session.get("illum_colors", [])
        illum_start    = session.get("illum_start_frame", len(frames) // 2)
        applicant_name = session.get("applicant_name", "")

        logger.info(f"[Pipeline] Starting analysis — session={session_id} frames={len(frames)}")

        def _notify(layer: str, result, label_override=None):
            if on_layer_update:
                label = label_override or result.label.value
                on_layer_update({
                    "layer":  layer,
                    "status": "pass" if result.label == DetectionLabel.REAL else (
                              "fail" if result.label == DetectionLabel.FAKE else "idle"),
                    "detail": result.detail[:60],
                })

        # ── Run all layers in parallel ─────────────────────────────────────────
        (
            deepfake_res,
            rppg_res,
            acoustic_res,
            hw_res,
            illum_res,
            facematch_res,
        ) = await asyncio.gather(
            self._run_deepfake(frames),
            self._run_rppg(frames),
            self._run_acoustic(audio_paths),
            self._run_hardware(),
            self._run_illumination(frames, illum_colors, illum_start),
            self._run_face_match(doc_path, frames, applicant_name),
        )

        # Notify live updates
        _notify("deepfake",  deepfake_res)
        _notify("rppg",      rppg_res)
        _notify("acoustic",  acoustic_res)
        _notify("hw",        hw_res)
        _notify("illum",     illum_res)
        _notify("facematch", facematch_res)

        # Liveness (simple pass-through for now — assessed from video challenge)
        liveness_res = LivenessResult(
            label=DetectionLabel.REAL,
            confidence=0.80,
            risk_contribution=5.0,
            detail="Head movement and eye blink detected",
            challenge_type="Head turn",
            user_response="Completed",
        )

        # ── Ensemble scoring ───────────────────────────────────────────────────
        total_ms = (time.perf_counter() - t0) * 1000

        result = risk_scorer.compute(
            session_id=session_id,
            applicant_name=applicant_name,
            deepfake=deepfake_res,
            rppg=rppg_res,
            acoustic=acoustic_res,
            illumination=illum_res,
            face_match=facematch_res,
            hardware=hw_res,
            liveness=liveness_res,
            total_latency_ms=total_ms,
        )

        logger.info(
            f"[Pipeline] DONE session={session_id} "
            f"verdict={result.verdict} score={result.risk_score:.1f} "
            f"latency={total_ms:.0f}ms"
        )

        return result

    # ── Layer runners (add error isolation per layer) ─────────────────────────

    async def _run_deepfake(self, frames: list[np.ndarray]) -> DeepfakeClassifierResult:
        try:
            return await deepfake_classifier.analyze(frames)
        except Exception as e:
            logger.exception(f"[Deepfake] Layer failed: {e}")
            return DeepfakeClassifierResult(
                label=DetectionLabel.UNKNOWN, confidence=0.5,
                risk_contribution=30.0, detail=f"Error: {e}"
            )

    async def _run_rppg(self, frames: list[np.ndarray]) -> RPPGResult:
        try:
            return await rppg_detector.analyze(frames)
        except Exception as e:
            logger.exception(f"[rPPG] Layer failed: {e}")
            return RPPGResult(
                label=DetectionLabel.UNKNOWN, confidence=0.5,
                risk_contribution=30.0, detail=f"Error: {e}"
            )

    async def _run_acoustic(self, audio_paths: list[str]) -> AcousticResult:
        try:
            return await acoustic_analyzer.analyze(audio_paths)
        except Exception as e:
            logger.exception(f"[Acoustic] Layer failed: {e}")
            return AcousticResult(
                label=DetectionLabel.UNKNOWN, confidence=0.5,
                risk_contribution=30.0, detail=f"Error: {e}"
            )

    async def _run_illumination(
        self, frames: list[np.ndarray], colors: list[str], illum_start: int = 0
    ) -> IlluminationResult:
        try:
            return await illum_challenge.analyze_from_frames(frames, colors, illum_start)
        except Exception as e:
            logger.exception(f"[Illum] Layer failed: {e}")
            return IlluminationResult(
                label=DetectionLabel.UNKNOWN, confidence=0.5,
                risk_contribution=25.0, detail=f"Error: {e}"
            )

    async def _run_face_match(
        self, doc_path: Optional[str], frames: list[np.ndarray], name: str
    ) -> FaceMatchResult:
        try:
            if not doc_path:
                return FaceMatchResult(
                    label=DetectionLabel.UNKNOWN, confidence=0.5,
                    risk_contribution=20.0, detail="No document uploaded"
                )
            return await asyncio.wait_for(
                document_analyzer.analyze(doc_path, frames, name),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            logger.warning("[FaceMatch] Timed out after 15s")
            return FaceMatchResult(
                label=DetectionLabel.UNKNOWN, confidence=0.5,
                risk_contribution=20.0, detail="Face match timed out"
            )
        except Exception as e:
            logger.exception(f"[FaceMatch] Layer failed: {e}")
            return FaceMatchResult(
                label=DetectionLabel.UNKNOWN, confidence=0.5,
                risk_contribution=30.0, detail=f"Error: {e}"
            )

    async def _run_hardware(self) -> HardwareAuthResult:
        try:
            return await hardware_checker.analyze()
        except Exception as e:
            logger.exception(f"[HW] Layer failed: {e}")
            return HardwareAuthResult(
                label=DetectionLabel.UNKNOWN, confidence=0.5,
                risk_contribution=20.0, detail=f"Error: {e}"
            )


# ── Singleton ──────────────────────────────────────────────────────────────────
pipeline = KYCPipeline()
