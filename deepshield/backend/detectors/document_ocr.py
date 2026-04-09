"""
backend/detectors/document_ocr.py
─────────────────────────────────────────────────────────
DeepShield KYC  –  Document OCR + ArcFace Face Matching

Two tasks:
  A) OCR: Extract name/DOB/ID number from uploaded document
          and verify it matches session data.
  B) Face Match: Compare face in document photo vs selfie
          using ArcFace (InsightFace) cosine similarity.
─────────────────────────────────────────────────────────
"""

import asyncio
import re
import time
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from loguru import logger

from backend.config import get_settings
from backend.models.schemas import FaceMatchResult, DetectionLabel

settings = get_settings()

# ── OCR backend ───────────────────────────────────────────────────────────────
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
    TESS_OK = True
except ImportError:
    TESS_OK = False

try:
    import easyocr
    EASY_OK = True
    _ocr_reader = None  # lazy init
except ImportError:
    EASY_OK = False
    _ocr_reader = None

# ── Face recognition backend ─────────────────────────────────────────────────
try:
    from insightface.app import FaceAnalysis
    _face_app = FaceAnalysis(name="buffalo_sc", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    _face_app.prepare(ctx_id=0, det_size=(640, 640))
    INSIGHT_OK = True
    logger.info("[FaceMatch] InsightFace (ArcFace) loaded")
except Exception:
    INSIGHT_OK = False
    logger.warning("[FaceMatch] InsightFace not available — using OpenCV face embeddings")

# DeepFace fallback
try:
    from deepface import DeepFace
    DEEPFACE_OK = True
    logger.info("[FaceMatch] DeepFace available as fallback")
except Exception:
    DEEPFACE_OK = False


class DocumentOCRAndFaceMatch:
    """OCR document text extraction + ArcFace face similarity."""

    # ── Public API ────────────────────────────────────────────────────────────

    async def analyze(
        self,
        doc_path: str,
        video_frames: list[np.ndarray],
        applicant_name: str,
    ) -> FaceMatchResult:
        t0 = time.perf_counter()

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._run, doc_path, video_frames, applicant_name
            )
        except Exception as e:
            logger.exception(f"[DocOCR] Error: {e}")
            result = FaceMatchResult(
                label=DetectionLabel.UNKNOWN,
                confidence=0.5,
                risk_contribution=30.0,
                detail=f"Document analysis error: {e}",
            )

        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    # ── Core analysis ─────────────────────────────────────────────────────────

    def _run(
        self,
        doc_path: str,
        video_frames: list[np.ndarray],
        applicant_name: str,
    ) -> FaceMatchResult:

        doc_img = self._load_doc_image(doc_path)
        if doc_img is None:
            return FaceMatchResult(
                label=DetectionLabel.UNKNOWN,
                confidence=0.5,
                risk_contribution=35.0,
                detail="Could not load Aadhaar/ID document image",
            )

        # ── Step A: OCR ───────────────────────────────────────────────────────
        ocr_text = self._run_ocr(doc_img)
        name_ok  = False
        if ocr_text and applicant_name:
            name_ok = self._name_in_text(applicant_name, ocr_text)
            logger.info(f"[DocOCR] OCR name match: {name_ok} | text: {ocr_text[:80]}")

        # ── Step B: Extract face from Aadhaar document ────────────────────────
        doc_face_emb = self._extract_face_embedding(doc_img)
        n_doc_faces  = 1 if doc_face_emb is not None else 0

        # ── Step C: Extract face from multiple video frames, take best match ──
        # Sample up to 5 frames for faster processing
        n_video_faces = 0
        cosine_sim    = 0.0
        best_sim      = 0.0

        if doc_face_emb is not None and video_frames:
            sample_indices = np.linspace(0, len(video_frames) - 1, min(5, len(video_frames)), dtype=int)
            for idx in sample_indices:
                frame_emb = self._extract_face_embedding(video_frames[idx])
                if frame_emb is not None:
                    n_video_faces += 1
                    sim = self._cosine_similarity(doc_face_emb, frame_emb)
                    if sim > best_sim:
                        best_sim = sim
            cosine_sim = best_sim
        elif doc_face_emb is None:
            best_frame = self._pick_best_frame(video_frames)
            if best_frame is not None:
                video_emb = self._extract_face_embedding(best_frame)
                if video_emb is not None:
                    n_video_faces = 1

        logger.info(f"[DocOCR] best_cosine={cosine_sim:.3f} name_ok={name_ok} "
                    f"doc_face={n_doc_faces} video_faces_sampled={n_video_faces}")

        # ── Decision: SAME PERSON or DIFFERENT PERSON ─────────────────────────
        face_match_ok = cosine_sim >= settings.FACE_MATCH_MIN_COSINE

        if doc_face_emb is None:
            label      = DetectionLabel.UNKNOWN
            risk       = 35.0
            confidence = 0.5
            detail     = "UNKNOWN — Could not extract face from Aadhaar document"
        elif n_video_faces == 0:
            label      = DetectionLabel.UNKNOWN
            risk       = 35.0
            confidence = 0.5
            detail     = "UNKNOWN — Could not detect face in live video"
        elif face_match_ok:
            label      = DetectionLabel.REAL
            risk       = max(0.0, (settings.FACE_MATCH_MIN_COSINE - cosine_sim) * 20)
            if not name_ok and ocr_text:
                risk  += 15.0
            confidence = min(0.97, 0.5 + (cosine_sim - settings.FACE_MATCH_MIN_COSINE) * 2)
            detail     = (f"SAME PERSON — Aadhaar face matches live video "
                          f"(similarity {cosine_sim:.2f}")
            detail    += ", name verified)" if name_ok else ", name OCR mismatch)"
        else:
            label      = DetectionLabel.FAKE
            risk       = 85.0
            confidence = min(0.95, 0.5 + (settings.FACE_MATCH_MIN_COSINE - cosine_sim))
            detail     = (f"DIFFERENT PERSON — Aadhaar face does NOT match live video "
                          f"(similarity {cosine_sim:.2f} < threshold {settings.FACE_MATCH_MIN_COSINE})")

        return FaceMatchResult(
            label=label,
            confidence=round(confidence, 3),
            risk_contribution=round(risk, 1),
            detail=detail,
            cosine_similarity=round(cosine_sim, 4),
            faces_detected_video=n_video_faces,
            faces_detected_doc=n_doc_faces,
        )

    # ── OCR helpers ───────────────────────────────────────────────────────────

    def _run_ocr(self, img: np.ndarray) -> Optional[str]:
        """Extract text from document image."""
        # Preprocess: grayscale + threshold for better OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if TESS_OK:
            try:
                text = pytesseract.image_to_string(thresh, config="--psm 6")
                return text.strip()
            except Exception as e:
                logger.warning(f"[OCR] Tesseract failed: {e}")

        if EASY_OK:
            try:
                global _ocr_reader
                if _ocr_reader is None:
                    _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                results = _ocr_reader.readtext(thresh)
                return " ".join([r[1] for r in results])
            except Exception as e:
                logger.warning(f"[OCR] EasyOCR failed: {e}")

        return None

    def _name_in_text(self, name: str, text: str) -> bool:
        """Check if applicant name appears in OCR text (fuzzy: last name match)."""
        name_parts = name.lower().split()
        text_lower = text.lower()
        # Match if any significant name part (len > 2) appears in text
        for part in name_parts:
            if len(part) > 2 and part in text_lower:
                return True
        return False

    # ── Face embedding helpers ────────────────────────────────────────────────

    def _extract_face_embedding(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding using InsightFace > DeepFace(Facenet512) > OpenCV fallback."""

        # Priority 1: InsightFace ArcFace (best accuracy)
        if INSIGHT_OK:
            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = _face_app.get(rgb)
                if faces:
                    best = max(faces, key=lambda f: f.det_score)
                    emb = best.embedding
                    return emb / (np.linalg.norm(emb) + 1e-8)
            except Exception as e:
                logger.warning(f"[FaceMatch] InsightFace error: {e}")

        # Priority 2: DeepFace with Facenet512 (proper identity embeddings)
        if DEEPFACE_OK:
            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = DeepFace.represent(
                    img_path=rgb,
                    model_name="Facenet",  # Facenet (128-d) faster than Facenet512
                    detector_backend="opencv",
                    enforce_detection=False,
                    align=True,
                )
                if result and len(result) > 0:
                    emb = np.array(result[0]["embedding"], dtype=np.float32)
                    return emb / (np.linalg.norm(emb) + 1e-8)
            except Exception as e:
                logger.warning(f"[FaceMatch] DeepFace error: {e}")

        # Priority 3: OpenCV Haar + LBP histogram (last resort, low accuracy)
        return self._cv_lbp_embedding(img)

    def _cv_lbp_embedding(self, img: np.ndarray) -> Optional[np.ndarray]:
        """LBP texture histogram on detected face region — better than color histogram."""
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2]*f[3])  # largest face
                # Add margin
                margin = int(0.1 * min(w, h))
                x1 = max(0, x - margin); y1 = max(0, y - margin)
                x2 = min(img.shape[1], x + w + margin)
                y2 = min(img.shape[0], y + h + margin)
                roi = gray[y1:y2, x1:x2]
            else:
                # Centre crop
                h_img, w_img = gray.shape
                roi = gray[h_img//4:3*h_img//4, w_img//4:3*w_img//4]

            roi = cv2.resize(roi, (128, 128))

            # Compute LBP manually
            lbp = np.zeros_like(roi)
            for i in range(1, roi.shape[0]-1):
                for j in range(1, roi.shape[1]-1):
                    center = roi[i, j]
                    code = 0
                    neighbors = [
                        roi[i-1,j-1], roi[i-1,j], roi[i-1,j+1],
                        roi[i,j+1], roi[i+1,j+1], roi[i+1,j],
                        roi[i+1,j-1], roi[i,j-1]
                    ]
                    for k, n in enumerate(neighbors):
                        code |= (1 << k) if n >= center else 0
                    lbp[i, j] = code

            # Histogram of LBP codes
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            emb = hist.astype(np.float32)
            return emb / (np.linalg.norm(emb) + 1e-8)
        except Exception:
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        return float(np.clip(np.dot(a_norm, b_norm), -1.0, 1.0))

    # ── Frame selection ────────────────────────────────────────────────────────

    def _pick_best_frame(self, frames: list[np.ndarray]) -> Optional[np.ndarray]:
        """Select the sharpest frame using Laplacian variance."""
        if not frames:
            return None
        best_score = -1
        best_frame = frames[0]
        for f in frames[::3]:    # sample every 3rd frame
            gray  = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if score > best_score:
                best_score = score
                best_frame = f
        return best_frame

    def _load_doc_image(self, path: str) -> Optional[np.ndarray]:
        """Load document: handle images and PDFs."""
        p = Path(path)
        if not p.exists():
            return None

        if p.suffix.lower() == ".pdf":
            try:
                from pdf2image import convert_from_path
                pages = convert_from_path(str(p), dpi=200, first_page=1, last_page=1)
                if pages:
                    import PIL.Image
                    img_array = np.array(pages[0])
                    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.warning(f"[DocOCR] PDF convert failed: {e}")
                return None

        img = cv2.imread(str(p))
        return img


# ── Singleton ──────────────────────────────────────────────────────────────────
document_analyzer = DocumentOCRAndFaceMatch()
