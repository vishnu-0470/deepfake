"""
backend/utils/scoring.py
─────────────────────────────────────────────────────────
DeepShield KYC  –  Weighted Ensemble Risk Scorer

Combines all layer outputs into a single 0–100 risk score
with explainability and verdict.

Weights (configurable in config.py):
  Deepfake classifier   30%
  rPPG                  25%
  Acoustic profiling    20%
  Illumination challenge 15%
  Face match            10%
  Hardware auth         +bonus/penalty (not weighted)
─────────────────────────────────────────────────────────
"""

from loguru import logger

from backend.config import get_settings
from backend.models.schemas import (
    KYCAnalysisResult, KYCVerdict, FraudType,
    DeepfakeClassifierResult, RPPGResult, AcousticResult,
    IlluminationResult, FaceMatchResult, HardwareAuthResult, LivenessResult,
    DetectionLabel,
)

settings = get_settings()


class RiskScorer:
    """Weighted ensemble risk scorer."""

    def compute(
        self,
        session_id: str,
        applicant_name: str,
        deepfake:     DeepfakeClassifierResult,
        rppg:         RPPGResult,
        acoustic:     AcousticResult,
        illumination: IlluminationResult,
        face_match:   FaceMatchResult,
        hardware:     HardwareAuthResult,
        liveness:     LivenessResult,
        total_latency_ms: float = 0.0,
    ) -> KYCAnalysisResult:

        # ── Weighted score ────────────────────────────────────────────────────
        weighted = (
            deepfake.risk_contribution     * settings.WEIGHT_DEEPFAKE_CLASSIFIER +
            rppg.risk_contribution         * settings.WEIGHT_RPPG               +
            acoustic.risk_contribution     * settings.WEIGHT_ACOUSTIC            +
            illumination.risk_contribution * settings.WEIGHT_ILLUMINATION        +
            face_match.risk_contribution   * settings.WEIGHT_FACE_MATCH
        )

        # Hardware penalty: virtual camera detected → add 20 points unconditionally
        hw_penalty = 20.0 if hardware.is_virtual else 0.0
        raw_score  = weighted + hw_penalty
        risk_score = round(min(max(raw_score, 0.0), 100.0), 1)

        # ── Verdict ───────────────────────────────────────────────────────────
        if risk_score >= settings.BLOCK_THRESHOLD:
            verdict = KYCVerdict.BLOCKED
        elif risk_score >= settings.REVIEW_THRESHOLD:
            verdict = KYCVerdict.REVIEW
        else:
            verdict = KYCVerdict.APPROVED

        # ── Fraud types ───────────────────────────────────────────────────────
        fraud_types = self._identify_fraud_types(
            deepfake, rppg, acoustic, illumination, face_match, hardware
        )
        if not fraud_types:
            fraud_types = [FraudType.NONE]

        # ── Explanation ───────────────────────────────────────────────────────
        explanation = self._build_explanation(verdict, risk_score, fraud_types,
                                               deepfake, rppg, face_match, hardware)

        logger.info(
            f"[Scorer] session={session_id} score={risk_score} "
            f"verdict={verdict} fraud={fraud_types}"
        )

        return KYCAnalysisResult(
            session_id=session_id,
            applicant_name=applicant_name,
            deepfake_result=deepfake,
            rppg_result=rppg,
            acoustic_result=acoustic,
            illumination_result=illumination,
            face_match_result=face_match,
            hardware_result=hardware,
            liveness_result=liveness,
            risk_score=risk_score,
            verdict=verdict,
            fraud_types=fraud_types,
            explanation=explanation,
            total_latency_ms=round(total_latency_ms, 1),
        )

    # ── Fraud classification ──────────────────────────────────────────────────

    def _identify_fraud_types(
        self,
        deepfake:     DeepfakeClassifierResult,
        rppg:         RPPGResult,
        acoustic:     AcousticResult,
        illumination: IlluminationResult,
        face_match:   FaceMatchResult,
        hardware:     HardwareAuthResult,
    ) -> list[FraudType]:
        types = []

        # Face-swap: deepfake + illumination both flagged
        if (deepfake.label == DetectionLabel.FAKE and
                illumination.label == DetectionLabel.FAKE):
            types.append(FraudType.FACE_SWAP)

        # GAN generated: deepfake + rPPG both flagged, no illumination response
        elif (deepfake.label == DetectionLabel.FAKE and
              rppg.label == DetectionLabel.FAKE):
            types.append(FraudType.GAN_GENERATED)

        # Only deepfake classifier flagged
        elif deepfake.label == DetectionLabel.FAKE:
            types.append(FraudType.FACE_SWAP)

        # Synthetic audio
        if acoustic.label == DetectionLabel.FAKE:
            types.append(FraudType.AUDIO_SYNTHETIC)

        # Virtual camera
        if hardware.is_virtual:
            types.append(FraudType.VIRTUAL_CAMERA)

        # ID mismatch (face match fail, but video might be real)
        if (face_match.label == DetectionLabel.FAKE and
                deepfake.label != DetectionLabel.FAKE):
            types.append(FraudType.ID_MISMATCH)

        # Liveness failure
        if rppg.label == DetectionLabel.FAKE and illumination.label == DetectionLabel.FAKE:
            if FraudType.FACE_SWAP not in types and FraudType.GAN_GENERATED not in types:
                types.append(FraudType.LIVENESS_FAIL)

        return list(dict.fromkeys(types))   # deduplicate, preserve order

    # ── Human-readable explanation ────────────────────────────────────────────

    def _build_explanation(
        self,
        verdict:    KYCVerdict,
        score:      float,
        fraud_types: list[FraudType],
        deepfake:   DeepfakeClassifierResult,
        rppg:       RPPGResult,
        face_match: FaceMatchResult,
        hardware:   HardwareAuthResult,
    ) -> str:
        if verdict == KYCVerdict.APPROVED:
            return (
                f"All {6} verification layers passed with a composite risk score of "
                f"{score}/100. Identity confirmed."
            )

        parts = [f"Risk score: {score}/100. "]
        if FraudType.FACE_SWAP in fraud_types:
            parts.append("Face-swap deepfake detected by video classifier and illumination challenge.")
        if FraudType.GAN_GENERATED in fraud_types:
            parts.append("GAN-generated synthetic face detected via FFT spectral fingerprint and rPPG absence.")
        if FraudType.AUDIO_SYNTHETIC in fraud_types:
            parts.append("Synthetic audio injection suspected (no room reverb, unnaturally high SNR).")
        if FraudType.VIRTUAL_CAMERA in fraud_types:
            parts.append(f"Virtual camera driver detected: {hardware.device_name}.")
        if FraudType.ID_MISMATCH in fraud_types:
            parts.append(f"Face does not match ID document (similarity {face_match.cosine_similarity:.2f}).")
        if not parts[1:]:
            parts.append("Multiple anomaly signals triggered manual review threshold.")

        return " ".join(parts)


# ── Singleton ──────────────────────────────────────────────────────────────────
risk_scorer = RiskScorer()
