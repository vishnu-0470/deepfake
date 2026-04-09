"""
backend/detectors/rppg_detector.py
─────────────────────────────────────────────────────────
DeepShield KYC  –  Remote Photoplethysmography (rPPG)

Deepfake overlays do not have beating hearts.
This module extracts the subtle color fluctuations in
facial skin caused by blood volume changes (heartbeat).

Algorithm: CHROM method (de Haan & Jeanne, 2013)
  1. Detect face with MediaPipe
  2. Extract mean RGB from forehead ROI each frame
  3. Project into chrominance channels
  4. Bandpass filter (0.75–4.0 Hz → 45–240 bpm)
  5. FFT → dominant frequency → heart rate
  6. If no frequency peak in physiological range → FAKE
─────────────────────────────────────────────────────────
"""

import asyncio
import time
from typing import Optional
import numpy as np
import cv2
from scipy import signal as scipy_signal
from loguru import logger

from backend.config import get_settings
from backend.models.schemas import RPPGResult, DetectionLabel

settings = get_settings()

try:
    import mediapipe as mp
    _mp_face = mp.solutions.face_detection
    MEDIAPIPE_OK = True
except (ImportError, AttributeError):
    MEDIAPIPE_OK = False
    logger.warning("[rPPG] MediaPipe not installed — face detection degraded")


class RPPGDetector:
    """CHROM-based rPPG detector."""

    def __init__(self):
        self._face_detector = None

    def _get_face_detector(self):
        if self._face_detector is None and MEDIAPIPE_OK:
            self._face_detector = _mp_face.FaceDetection(
                model_selection=0, min_detection_confidence=0.7
            )
        return self._face_detector

    # ── Public API ────────────────────────────────────────────────────────────

    async def analyze(self, frames: list[np.ndarray]) -> RPPGResult:
        t0 = time.perf_counter()

        if len(frames) < 20:
            return RPPGResult(
                label=DetectionLabel.UNKNOWN,
                confidence=0.5,
                risk_contribution=30.0,
                detail=f"Insufficient frames ({len(frames)} < 20 required)",
                latency_ms=0,
            )

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._run_chrom, frames
            )
        except Exception as e:
            logger.exception(f"[rPPG] Error: {e}")
            result = RPPGResult(
                label=DetectionLabel.UNKNOWN,
                confidence=0.5,
                risk_contribution=30.0,
                detail=f"Analysis error: {e}",
            )

        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    async def quick_check(self, frames: list[np.ndarray]) -> Optional[bool]:
        """Fast partial check for live streaming updates. Returns True=likely real, False=likely fake, None=uncertain."""
        if len(frames) < 30:
            return None
        try:
            rgb_signals = self._extract_rgb_signals(frames[:30])
            if rgb_signals is None or len(rgb_signals) < 25:
                return None
            variance = float(np.var(rgb_signals[:, 1]))   # green channel variance
            return variance > 0.1
        except Exception:
            return None

    # ── Core CHROM algorithm ─────────────────────────────────────────────────

    def _run_chrom(self, frames: list[np.ndarray]) -> RPPGResult:
        # Estimate actual FPS from frame count (recorded over ~8-30s)
        fps = max(5, min(settings.RPPG_FPS, len(frames) // 8))

        # Step 1: Extract raw RGB signals from forehead ROI
        rgb_signals = self._extract_rgb_signals(frames)
        if rgb_signals is None or len(rgb_signals) < 15:
            return RPPGResult(
                label=DetectionLabel.FAKE,
                confidence=0.72,
                risk_contribution=65.0,
                detail="Could not extract stable face ROI from video",
                signal_strength=0.0,
            )

        n = len(rgb_signals)
        R = rgb_signals[:, 0].astype(np.float64)
        G = rgb_signals[:, 1].astype(np.float64)
        B = rgb_signals[:, 2].astype(np.float64)

        # Step 2: Normalize (remove DC component)
        Rn = R / (np.mean(R) + 1e-8)
        Gn = G / (np.mean(G) + 1e-8)
        Bn = B / (np.mean(B) + 1e-8)

        # Step 3: CHROM projection
        Xs = 3 * Rn - 2 * Gn
        Ys = 1.5 * Rn + Gn - 1.5 * Bn

        # Step 4: Build rPPG signal
        alpha = np.std(Xs) / (np.std(Ys) + 1e-8)
        rppg_raw = Xs - alpha * Ys

        # Step 5: Bandpass filter (0.75–4.0 Hz)
        nyq    = fps / 2.0
        low    = settings.RPPG_MIN_HZ / nyq
        high   = min(settings.RPPG_MAX_HZ / nyq, 0.99)
        try:
            b, a   = scipy_signal.butter(4, [low, high], btype="bandpass")
            rppg   = scipy_signal.filtfilt(b, a, rppg_raw)
        except Exception:
            rppg = rppg_raw

        # Step 6: FFT → dominant frequency
        freqs = np.fft.rfftfreq(n, d=1.0 / fps)
        fft   = np.abs(np.fft.rfft(rppg))

        mask  = (freqs >= settings.RPPG_MIN_HZ) & (freqs <= settings.RPPG_MAX_HZ)
        fft_bio = fft[mask]
        freqs_bio = freqs[mask]

        if len(fft_bio) == 0:
            dominant_freq = None
            bpm = None
            peak_power = 0.0
        else:
            peak_idx = int(np.argmax(fft_bio))
            dominant_freq = float(freqs_bio[peak_idx])
            bpm = dominant_freq * 60.0
            peak_power = float(fft_bio[peak_idx])

        # Step 7: Signal quality (SNR proxy)
        total_power = float(np.sum(fft_bio) + 1e-8) if len(fft_bio) else 1.0
        signal_ratio = peak_power / total_power if total_power else 0.0

        # Step 8: Balanced decision
        bpm_ok        = (bpm is not None) and (45 <= bpm <= 200)
        snr_ok        = signal_ratio > 0.12   # balanced: catches photos but not real faces
        variance_ok   = float(np.var(rppg)) > 5e-7 if len(rppg) > 0 else False
        is_real       = bpm_ok and snr_ok and variance_ok
        confidence    = min(0.95, signal_ratio * 3.0) if is_real else max(0.55, 1.0 - signal_ratio * 2)

        if is_real:
            label = DetectionLabel.REAL
            risk  = max(0.0, 15.0 - signal_ratio * 20)
            detail = f"Heart rate detected: {bpm:.1f} bpm (SNR={signal_ratio:.3f})"
        else:
            label = DetectionLabel.FAKE
            risk  = 75.0
            if bpm is None or not variance_ok:
                detail = f"No pulse signal detected (SNR={signal_ratio:.4f}) — possible photo/static image"
            elif not bpm_ok:
                detail = f"BPM {bpm:.1f} outside physiological range"
            else:
                detail = f"Weak biological signal (SNR={signal_ratio:.3f})"

        logger.info(f"[rPPG] label={label} BPM={bpm} snr={signal_ratio:.3f} conf={confidence:.2f}")

        return RPPGResult(
            label=label,
            confidence=round(confidence, 3),
            risk_contribution=round(risk, 1),
            detail=detail,
            estimated_bpm=round(bpm, 1) if bpm else None,
            dominant_frequency=round(dominant_freq, 3) if dominant_freq else None,
            signal_strength=round(signal_ratio, 4),
            frames_analyzed=n,
        )

    # ── Face ROI extraction ───────────────────────────────────────────────────

    def _extract_rgb_signals(self, frames: list[np.ndarray]) -> Optional[np.ndarray]:
        """Return (N, 3) array of mean RGB from forehead ROI per frame."""
        detector = self._get_face_detector()
        signals  = []

        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Try MediaPipe first
            if detector:
                results = detector.process(rgb)
                if results.detections:
                    det  = results.detections[0]
                    bbox = det.location_data.relative_bounding_box
                    h, w = frame.shape[:2]
                    x1   = int(bbox.xmin * w)
                    y1   = int(bbox.ymin * h)
                    fw   = int(bbox.width * w)
                    fh   = int(bbox.height * h)

                    # Forehead = top 20% of face bbox
                    forehead_y1 = max(0, y1)
                    forehead_y2 = max(0, y1 + int(fh * 0.2))
                    forehead_x1 = max(0, x1 + int(fw * 0.2))
                    forehead_x2 = min(w, x1 + int(fw * 0.8))

                    roi = rgb[forehead_y1:forehead_y2, forehead_x1:forehead_x2]
                    if roi.size > 0:
                        signals.append(roi.mean(axis=(0, 1)))
                        continue

            # Fallback: center crop as face proxy
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            roi = rgb[cy - 60: cy - 30, cx - 40: cx + 40]
            if roi.size > 0:
                signals.append(roi.mean(axis=(0, 1)))

        return np.array(signals) if signals else None


# ── Singleton ──────────────────────────────────────────────────────────────────
rppg_detector = RPPGDetector()
