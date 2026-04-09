"""
backend/detectors/illumination_challenge.py
─────────────────────────────────────────────────────────
DeepShield KYC  –  Screen Illumination Challenge

A physical human face accurately reflects sudden screen
color flashes in real-time. A pre-rendered deepfake or
live face-swap model CANNOT adapt its lighting to sudden
screen changes — exposing the fraud.

Algorithm:
  1. Before challenge: record baseline facial color mean
  2. During each color flash: record facial color mean
  3. Compute per-channel color shift correlation between
     expected flash color and observed face color change
  4. If correlation < threshold → FAKE
─────────────────────────────────────────────────────────
"""

import asyncio
import time
from typing import Optional
import numpy as np
import cv2
from loguru import logger

from backend.config import get_settings
from backend.models.schemas import IlluminationResult, DetectionLabel

settings = get_settings()

try:
    import mediapipe as mp
    _mp_face_mesh = mp.solutions.face_mesh
    MEDIAPIPE_OK = True
except (ImportError, AttributeError):
    MEDIAPIPE_OK = False


# Map hex color strings to normalized BGR channels
def _hex_to_bgr_norm(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return np.array([b, g, r], dtype=np.float32) / 255.0


class IlluminationChallenge:
    """Validates that the face reflects screen illumination changes."""

    def __init__(self):
        self._face_mesh = None

    def _get_mesh(self):
        if self._face_mesh is None and MEDIAPIPE_OK:
            self._face_mesh = _mp_face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                min_detection_confidence=0.5,
            )
        return self._face_mesh

    # ── Public API ────────────────────────────────────────────────────────────

    async def analyze(
        self,
        frames_before: list[np.ndarray],
        frames_during: dict[str, list[np.ndarray]],
        challenge_colors: list[str],
    ) -> IlluminationResult:
        """
        frames_before : frames captured BEFORE the flash challenge
        frames_during : dict mapping hex_color → frames captured DURING that flash
        challenge_colors: ordered list of flash colors used
        """
        t0 = time.perf_counter()

        if not frames_before or not frames_during:
            return IlluminationResult(
                label=DetectionLabel.UNKNOWN,
                confidence=0.5,
                risk_contribution=30.0,
                detail="Insufficient challenge frames captured",
                latency_ms=0,
            )

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._run_analysis, frames_before, frames_during, challenge_colors
            )
        except Exception as e:
            logger.exception(f"[Illum] Error: {e}")
            result = IlluminationResult(
                label=DetectionLabel.UNKNOWN,
                confidence=0.5,
                risk_contribution=30.0,
                detail=f"Challenge analysis error: {e}",
            )

        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    async def analyze_from_frames(
        self,
        all_frames: list[np.ndarray],
        challenge_colors: list[str],
        challenge_start_frame: int = 0,
    ) -> IlluminationResult:
        """
        Simplified version when we don't have frame-level timing:
        splits frames into pre-challenge and post-challenge halves.
        """
        n = len(all_frames)
        if n < 10:
            return IlluminationResult(
                label=DetectionLabel.UNKNOWN,
                confidence=0.5,
                risk_contribution=30.0,
                detail="Not enough frames for illumination analysis",
            )

        split = challenge_start_frame if challenge_start_frame > 0 else n // 2
        before = all_frames[:split]
        after  = all_frames[split:]

        # Simulate per-color segments
        seg_len = max(1, len(after) // len(challenge_colors))
        frames_during = {}
        for i, color in enumerate(challenge_colors):
            start = i * seg_len
            end   = start + seg_len
            frames_during[color] = after[start:end] if start < len(after) else after[-1:]

        return await self.analyze(before, frames_during, challenge_colors)

    # ── Core analysis ─────────────────────────────────────────────────────────

    def _run_analysis(
        self,
        frames_before: list[np.ndarray],
        frames_during: dict[str, list[np.ndarray]],
        challenge_colors: list[str],
    ) -> IlluminationResult:

        # Step 1: Baseline facial color (mean across pre-challenge frames)
        baseline_color = self._mean_face_color(frames_before)
        if baseline_color is None:
            return IlluminationResult(
                label=DetectionLabel.UNKNOWN,
                confidence=0.5,
                risk_contribution=40.0,
                detail="Could not detect face in baseline frames",
                challenges_sent=len(challenge_colors),
            )

        passed  = 0
        total   = 0
        correlations = []

        for hex_color in challenge_colors:
            flash_frames = frames_during.get(hex_color, [])
            if not flash_frames:
                continue

            total += 1
            during_color = self._mean_face_color(flash_frames)
            if during_color is None:
                continue

            # Step 2: Expected color shift direction
            expected_shift = _hex_to_bgr_norm(hex_color)     # illumination source
            observed_shift = during_color - baseline_color    # actual face change

            # Normalize observed shift
            obs_norm = observed_shift / (np.linalg.norm(observed_shift) + 1e-8)
            exp_norm = expected_shift / (np.linalg.norm(expected_shift) + 1e-8)

            # Step 3: Cosine similarity between expected and observed shift
            corr = float(np.dot(obs_norm, exp_norm))
            correlations.append(corr)

            if corr >= settings.ILLUM_CORRELATION_THRESHOLD:
                passed += 1

        if not correlations:
            return IlluminationResult(
                label=DetectionLabel.UNKNOWN,
                confidence=0.5,
                risk_contribution=35.0,
                detail="No challenge responses captured",
                challenges_sent=len(challenge_colors),
            )

        mean_corr = float(np.mean(correlations))
        pass_rate = passed / total if total else 0.0

        # Balanced decision
        # Real face: has response AND some variation between colors
        # Photo: high uniform response (low variance)
        # Deepfake: no response
        has_response  = mean_corr >= 0.15
        has_variation = corr_variance >= 0.005  # lowered to avoid false positives
        is_real       = has_response and (has_variation or pass_rate >= 0.5)

        confidence = min(0.90, 0.5 + abs(mean_corr) * 0.6)
        risk_score = 0.0 if is_real else max(35.0, (0.5 - mean_corr) * 70)
        risk_score = min(risk_score, 85.0)

        if is_real:
            label  = DetectionLabel.REAL
            detail = f"Face illumination response confirmed ({passed}/{total} passed, corr={mean_corr:.2f})"
        else:
            label  = DetectionLabel.FAKE
            if not has_response:
                detail = f"No illumination response — possible deepfake (corr={mean_corr:.2f})"
            else:
                detail = f"Uniform reflection — possible printed photo (var={corr_variance:.4f})"

        logger.info(f"[Illum] label={label} corr={mean_corr:.2f} passed={passed}/{total}")

        return IlluminationResult(
            label=label,
            confidence=round(confidence, 3),
            risk_contribution=round(risk_score, 1),
            detail=detail,
            challenges_sent=len(challenge_colors),
            challenges_passed=passed,
            correlation_score=round(mean_corr, 4),
        )

    # ── Face color extraction ─────────────────────────────────────────────────

    def _mean_face_color(self, frames: list[np.ndarray]) -> Optional[np.ndarray]:
        """Return mean BGR color of the face region across frames."""
        colors = []
        mesh   = self._get_mesh()

        for frame in frames:
            color = self._extract_face_mean(frame, mesh)
            if color is not None:
                colors.append(color)

        if not colors:
            return None
        return np.mean(colors, axis=0)

    def _extract_face_mean(
        self, frame: np.ndarray, mesh
    ) -> Optional[np.ndarray]:
        """Extract mean BGR color from facial skin region."""
        h, w = frame.shape[:2]

        if mesh:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                # Forehead + cheek landmarks for stable color extraction
                key_idxs = [10, 338, 297, 332, 284, 54, 103, 67, 109]
                pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in key_idxs])

                # Create face mask from convex hull
                mask = np.zeros((h, w), dtype=np.uint8)
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(mask, hull, 255)

                mean_val = cv2.mean(frame, mask=mask)[:3]
                return np.array(mean_val, dtype=np.float32) / 255.0

        # Fallback: centre crop
        cy, cx = h // 2, w // 2
        roi = frame[cy - 50:cy + 50, cx - 40:cx + 40]
        if roi.size == 0:
            return None
        return np.array(roi.mean(axis=(0, 1)), dtype=np.float32) / 255.0


# ── Singleton ──────────────────────────────────────────────────────────────────
illum_challenge = IlluminationChallenge()
