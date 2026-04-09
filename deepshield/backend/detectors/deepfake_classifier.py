"""
backend/detectors/deepfake_classifier.py
─────────────────────────────────────────────────────────
DeepShield KYC  –  Deepfake Frame Classifier

Two-stage detection:
  Stage A: EfficientNet-B4 fine-tuned on FaceForensics++
           Detects blending boundaries, GAN artifacts,
           texture inconsistencies.
  Stage B: 2D FFT spectral fingerprint analysis
           GAN-generated images leave a characteristic
           spectral grid pattern invisible to the human eye.
─────────────────────────────────────────────────────────
"""

import asyncio
import time
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from loguru import logger

from backend.config import get_settings
from backend.models.schemas import DeepfakeClassifierResult, DetectionLabel

settings = get_settings()

# ── Try importing PyTorch ─────────────────────────────────────────────────────
try:
    import torch
    import torchvision.transforms as T
    import timm
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    logger.warning("[Classifier] PyTorch/timm not installed — using FFT-only mode")

try:
    import mediapipe as mp
    _mp_fd = mp.solutions.face_detection
    MP_OK = True
except (ImportError, AttributeError):
    MP_OK = False


# ── Model transforms ──────────────────────────────────────────────────────────
_TRANSFORM = None
if TORCH_OK:
    _TRANSFORM = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class DeepfakeClassifier:
    """EfficientNet-B4 + FFT deepfake detector."""

    def __init__(self):
        self._model  = None
        self._device = None
        self._face_detector = None

    # ── Warmup ───────────────────────────────────────────────────────────────

    def load_model(self):
        if not TORCH_OK:
            return
        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = timm.create_model(
                settings.CLASSIFIER_MODEL_NAME,
                pretrained=True,
                num_classes=1,          # binary: real vs fake
            )
            # Load fine-tuned weights if checkpoint exists
            ckpt_path = settings.MODEL_DIR / settings.CLASSIFIER_CHECKPOINT
            if Path(ckpt_path).exists():
                state = torch.load(ckpt_path, map_location=self._device)
                model.load_state_dict(state, strict=False)
                logger.info(f"[Classifier] Loaded checkpoint: {ckpt_path}")
            else:
                logger.warning(f"[Classifier] Checkpoint not found ({ckpt_path}) — using pretrained weights")

            model.eval()
            model.to(self._device)
            self._model = model
            logger.info(f"[Classifier] EfficientNet-B4 loaded on {self._device}")
        except Exception as e:
            logger.exception(f"[Classifier] Model load failed: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    async def analyze(self, frames: list[np.ndarray]) -> DeepfakeClassifierResult:
        t0 = time.perf_counter()

        if not frames:
            return DeepfakeClassifierResult(
                label=DetectionLabel.UNKNOWN,
                confidence=0.5,
                risk_contribution=30.0,
                detail="No frames to analyze",
                latency_ms=0,
            )

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._run_detection, frames
            )
        except Exception as e:
            logger.exception(f"[Classifier] Error: {e}")
            result = DeepfakeClassifierResult(
                label=DetectionLabel.UNKNOWN,
                confidence=0.5,
                risk_contribution=30.0,
                detail=f"Detection error: {e}",
            )

        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    # ── Core detection ────────────────────────────────────────────────────────

    def _run_detection(self, frames: list[np.ndarray]) -> DeepfakeClassifierResult:
        # Sample up to 8 frames for faster processing
        indices = np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int)
        sampled = [frames[i] for i in indices]

        # Extra check: detect if video is static (printed photo held to camera)
        if self._is_static_image(frames):
            return DeepfakeClassifierResult(
                label=DetectionLabel.FAKE,
                confidence=0.95,
                risk_contribution=90.0,
                detail="Static image detected — no motion/depth variation across frames",
                frames_analyzed=len(frames),
            )

        # Crop face from each frame
        face_crops = [self._crop_face(f) for f in sampled]
        face_crops = [f for f in face_crops if f is not None]

        if not face_crops:
            return DeepfakeClassifierResult(
                label=DetectionLabel.UNKNOWN,
                confidence=0.5,
                risk_contribution=30.0,
                detail="No face detected in sampled frames",
                frames_analyzed=len(sampled),
            )

        # Stage A: Neural network classification
        nn_scores  = self._nn_classify(face_crops)

        # Stage B: FFT spectral analysis
        fft_scores = [self._fft_fake_score(crop) for crop in face_crops]

        # Combine scores
        nn_mean  = float(np.mean(nn_scores))   if nn_scores  else 0.5
        fft_mean = float(np.mean(fft_scores))  if fft_scores else 0.0

        # If NN available: weighted combination; else FFT-only
        if nn_scores:
            fake_score = 0.70 * nn_mean + 0.30 * fft_mean
        else:
            fake_score = 0.40 + fft_mean * 0.60   # FFT alone gives partial signal

        fake_score    = float(np.clip(fake_score, 0.0, 1.0))
        top_fake      = float(np.max(nn_scores)) if nn_scores else fake_score
        is_fake       = fake_score >= settings.CLASSIFIER_FAKE_THRESHOLD
        confidence    = min(0.97, abs(fake_score - 0.5) * 2 + 0.5)
        risk_score    = fake_score * 95.0

        if is_fake:
            label  = DetectionLabel.FAKE
            detail = (
                f"Deepfake score: {fake_score:.2f} "
                f"(NN:{nn_mean:.2f}, FFT:{fft_mean:.2f})"
            )
        else:
            label  = DetectionLabel.REAL
            detail = (
                f"Authenticity score: {1-fake_score:.2f} "
                f"(NN:{1-nn_mean:.2f}, FFT:{1-fft_mean:.2f})"
            )

        logger.info(f"[Classifier] label={label} fake_score={fake_score:.3f} "
                    f"top={top_fake:.3f} frames={len(face_crops)}")

        return DeepfakeClassifierResult(
            label=label,
            confidence=round(confidence, 3),
            risk_contribution=round(risk_score, 1),
            detail=detail,
            frames_analyzed=len(face_crops),
            top_fake_frame_score=round(top_fake, 3),
        )

    # ── Static image detection ────────────────────────────────────────────────

    def _is_static_image(self, frames: list[np.ndarray]) -> bool:
        """Detect if video is a static photo held to camera."""
        if len(frames) < 10:
            return False
        try:
            indices = np.linspace(0, len(frames)-1, min(10, len(frames)), dtype=int)
            grays = [cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32) for i in indices]
            diffs = [np.mean(np.abs(grays[i] - grays[i-1])) for i in range(1, len(grays))]
            mean_diff = float(np.mean(diffs))
            max_diff  = float(np.max(diffs))
            logger.info(f"[Classifier] Frame motion: mean={mean_diff:.3f} max={max_diff:.3f}")
            # Only flag as static if VERY still — real faces always have some motion
            # Threshold lowered to avoid false positives on slow-moving real faces
            return mean_diff < 0.15 and max_diff < 0.8
        except Exception:
            return False

    # ── Neural network ────────────────────────────────────────────────────────

    def _nn_classify(self, crops: list[np.ndarray]) -> list[float]:
        """Run EfficientNet-B4 on face crops. Returns list of fake probabilities."""
        if not TORCH_OK or self._model is None:
            return []
        scores = []
        with torch.no_grad():
            for crop in crops:
                rgb   = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = _TRANSFORM(rgb).unsqueeze(0).to(self._device)
                logit  = self._model(tensor)
                prob   = torch.sigmoid(logit).item()
                scores.append(float(prob))
        return scores

    # ── FFT spectral fingerprint ──────────────────────────────────────────────

    def _fft_fake_score(self, crop: np.ndarray) -> float:
        """
        GAN-generated images leave a characteristic grid pattern in the
        frequency domain due to upsampling artifacts. Returns a score
        0=clean, 1=strong GAN fingerprint.
        """
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray = cv2.resize(gray, (256, 256))

            # 2D FFT + shift to centre
            fft     = np.fft.fft2(gray)
            fft_mag = np.log1p(np.abs(np.fft.fftshift(fft)))

            # Normalize
            fft_norm = (fft_mag - fft_mag.min()) / (fft_mag.max() - fft_mag.min() + 1e-8)

            # Look for repeating grid pattern in mid-frequency range
            h, w = fft_norm.shape
            cy, cx = h // 2, w // 2

            # Sample mid-frequency ring (radius 20–80 pixels from centre)
            y_idx, x_idx = np.ogrid[:h, :w]
            dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
            ring_mask = (dist >= 20) & (dist <= 80)
            ring_vals = fft_norm[ring_mask]

            # GAN artifact: unusually high kurtosis (spike) in mid-frequency ring
            if len(ring_vals) < 10:
                return 0.0

            mean = np.mean(ring_vals)
            std  = np.std(ring_vals) + 1e-8
            kurtosis = np.mean(((ring_vals - mean) / std) ** 4)

            # Peak-to-mean ratio in ring
            peak_mean_ratio = np.max(ring_vals) / (mean + 1e-8)

            # Score: both high kurtosis AND high peak-to-mean → GAN artifact
            raw_score = (min(kurtosis, 20.0) / 20.0) * 0.5 + (min(peak_mean_ratio, 5.0) / 5.0) * 0.5
            return float(np.clip(raw_score, 0.0, 1.0))

        except Exception:
            return 0.0

    # ── Face crop ─────────────────────────────────────────────────────────────

    def _crop_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Crop and return face region. Falls back to centre crop."""
        if MP_OK:
            if self._face_detector is None:
                self._face_detector = _mp_fd.FaceDetection(
                    model_selection=0, min_detection_confidence=0.6
                )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self._face_detector.process(rgb)
            if res.detections:
                det  = res.detections[0]
                bbox = det.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                x2 = min(w, x1 + int(bbox.width * w))
                y2 = min(h, y1 + int(bbox.height * h))
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    return cv2.resize(crop, (224, 224))

        # Centre fallback
        h, w = frame.shape[:2]
        size  = min(h, w)
        cy, cx = h // 2, w // 2
        half   = size // 2
        crop   = frame[cy - half:cy + half, cx - half:cx + half]
        return cv2.resize(crop, (224, 224)) if crop.size > 0 else None


# ── Singleton ──────────────────────────────────────────────────────────────────
deepfake_classifier = DeepfakeClassifier()
