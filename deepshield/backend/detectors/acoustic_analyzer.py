"""
backend/detectors/acoustic_analyzer.py
─────────────────────────────────────────────────────────
DeepShield KYC  –  Environmental Acoustic & Noise Profiling

AI-generated voice cloning produces audio that is too clean.
A real human in a physical room generates:
  • Ambient background noise
  • Room reverberation (echo / RT60)
  • Natural SNR in the 15–45 dB range

Algorithm:
  1. Convert WebM → WAV (ffmpeg)
  2. Separate speech segments (VAD)
  3. Estimate SNR (signal-to-noise ratio)
  4. Estimate RT60 via backward integration (Schroeder method)
  5. Detect presence of natural room noise fingerprint
  6. Flag if audio is too clean (cloned/synthetic)
─────────────────────────────────────────────────────────
"""

import asyncio
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import numpy as np
from scipy import signal as scipy_signal
from scipy.io import wavfile
from loguru import logger

from backend.config import get_settings
from backend.models.schemas import AcousticResult, DetectionLabel

settings = get_settings()

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False
    logger.warning("[Acoustic] librosa not installed — using scipy fallback")


class AcousticAnalyzer:
    """Environmental acoustic profiling for deepfake audio detection."""

    # ── Public API ────────────────────────────────────────────────────────────

    async def analyze(self, audio_paths: list[str]) -> AcousticResult:
        t0 = time.perf_counter()

        if not audio_paths:
            return AcousticResult(
                label=DetectionLabel.UNKNOWN,
                confidence=0.5,
                risk_contribution=30.0,
                detail="No audio data available for analysis",
                latency_ms=0,
            )

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._run_analysis, audio_paths
            )
        except Exception as e:
            logger.exception(f"[Acoustic] Error: {e}")
            result = AcousticResult(
                label=DetectionLabel.UNKNOWN,
                confidence=0.5,
                risk_contribution=30.0,
                detail=f"Acoustic analysis error: {e}",
            )

        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    # ── Core analysis ─────────────────────────────────────────────────────────

    def _run_analysis(self, audio_paths: list[str]) -> AcousticResult:
        # Step 1: Decode all WebM files to float32 WAV arrays
        samples_all = []
        sr = settings.ACOUSTIC_SAMPLE_RATE

        for path in audio_paths:
            audio = self._decode_audio(path, sr)
            if audio is not None:
                samples_all.append(audio)

        if not samples_all:
            return AcousticResult(
                label=DetectionLabel.UNKNOWN,
                confidence=0.5,
                risk_contribution=40.0,
                detail="Could not decode any audio files",
            )

        audio = np.concatenate(samples_all)

        # Step 2: Estimate SNR
        snr_db = self._estimate_snr(audio, sr)

        # Step 3: Estimate RT60 (room reverberation)
        rt60 = self._estimate_rt60(audio, sr)

        # Step 4: Spectral flatness (noise texture)
        spectral_flatness = self._spectral_flatness(audio, sr)

        # Step 5: Zero-crossing rate variance (too steady = synthetic)
        zcr_variance = self._zcr_variance(audio, sr)

        # ── Decision logic ───────────────────────────────────────────────────
        # Synthetic audio traits:
        #   • Very high SNR (>40 dB) — no background noise
        #   • No reverb (RT60 < 0.05s)
        #   • Low spectral flatness variance (too uniform)

        risk_score = 0.0
        signals = []

        # SNR check
        if snr_db is not None:
            if snr_db > 40.0:
                risk_score += 35.0
                signals.append(f"SNR too high: {snr_db:.1f} dB (no ambient noise)")
            elif snr_db < settings.ACOUSTIC_MIN_SNR_DB:
                risk_score += 5.0   # very noisy — still suspicious but less so

        # RT60 check
        has_reverb = False
        if rt60 is not None:
            has_reverb = rt60 >= settings.ACOUSTIC_REVERB_MIN_RT60
            if not has_reverb:
                risk_score += 30.0
                signals.append(f"No room reverb detected (RT60={rt60*1000:.0f} ms)")

        # Spectral flatness check
        if spectral_flatness is not None and spectral_flatness < 0.05:
            risk_score += 20.0
            signals.append("Unnaturally uniform spectral texture")

        # ZCR variance check
        if zcr_variance is not None and zcr_variance < 0.001:
            risk_score += 15.0
            signals.append("Zero-crossing rate too stable (synthetic audio pattern)")

        risk_score = min(risk_score, 100.0)
        is_real    = risk_score < 40.0
        confidence = 0.55 + abs(risk_score - 50) / 100.0
        confidence = min(confidence, 0.97)

        if is_real:
            label  = DetectionLabel.REAL
            detail = f"Natural acoustic environment — SNR {snr_db:.1f} dB"
            if has_reverb:
                detail += f", RT60 {rt60*1000:.0f} ms"
        else:
            label  = DetectionLabel.FAKE
            detail = " | ".join(signals) if signals else "Synthetic audio detected"

        logger.info(f"[Acoustic] label={label} SNR={snr_db} RT60={rt60} risk={risk_score:.1f}")

        return AcousticResult(
            label=label,
            confidence=round(confidence, 3),
            risk_contribution=round(risk_score, 1),
            detail=detail,
            estimated_snr_db=round(snr_db, 1) if snr_db is not None else None,
            rt60_ms=round(rt60 * 1000, 1) if rt60 is not None else None,
            has_reverb=has_reverb,
        )

    # ── Audio decode ─────────────────────────────────────────────────────────

    def _decode_audio(self, path: str, target_sr: int) -> Optional[np.ndarray]:
        """Decode audio file to float32 numpy array. Tries ffmpeg first, then soundfile."""
        try:
            # Try soundfile directly first (works on webm/ogg if codec available)
            try:
                import soundfile as sf
                audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if LIBROSA_OK and file_sr != target_sr:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=file_sr, target_sr=target_sr)
                return audio.astype(np.float32)
            except Exception:
                pass

            # Try librosa directly (handles many formats)
            if LIBROSA_OK:
                try:
                    import librosa
                    audio, _ = librosa.load(path, sr=target_sr, mono=True)
                    return audio
                except Exception:
                    pass

            # Try ffmpeg conversion
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp_path = tf.name

            result = subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ar", str(target_sr),
                 "-ac", "1", "-f", "wav", tmp_path],
                capture_output=True, timeout=30
            )
            if result.returncode == 0:
                if LIBROSA_OK:
                    import librosa
                    audio, _ = librosa.load(tmp_path, sr=target_sr, mono=True)
                else:
                    _, audio = wavfile.read(tmp_path)
                    audio = audio.astype(np.float32) / 32768.0
                Path(tmp_path).unlink(missing_ok=True)
                return audio

            Path(tmp_path).unlink(missing_ok=True)
            return None

        except Exception as e:
            logger.warning(f"[Acoustic] Decode failed for {path}: {e}")
            return None

    # ── Signal metrics ────────────────────────────────────────────────────────

    def _estimate_snr(self, audio: np.ndarray, sr: int) -> Optional[float]:
        """Estimate SNR using voice activity detection heuristic."""
        try:
            frame_len = int(sr * 0.02)  # 20ms frames
            frames = np.array_split(audio, max(1, len(audio) // frame_len))

            energies = np.array([np.sqrt(np.mean(f ** 2)) for f in frames if len(f) > 0])
            if len(energies) < 4:
                return None

            energies_sorted = np.sort(energies)
            noise_floor   = np.mean(energies_sorted[:max(1, len(energies_sorted) // 10)])
            signal_level  = np.mean(energies_sorted[int(len(energies_sorted) * 0.5):])

            if noise_floor < 1e-9:
                return 60.0  # essentially silent background → very high SNR (suspicious)

            snr = 20 * np.log10(signal_level / noise_floor + 1e-9)
            return float(np.clip(snr, 0, 80))
        except Exception:
            return None

    def _estimate_rt60(self, audio: np.ndarray, sr: int) -> Optional[float]:
        """
        Estimate room RT60 using the Schroeder backward-integration method.
        RT60: time for sound to decay 60 dB after source stops.
        """
        try:
            # Find the peak and take a window after it
            peak_idx = int(np.argmax(np.abs(audio)))
            decay    = audio[peak_idx:]

            if len(decay) < sr * 0.1:
                return 0.0

            # Energy decay curve
            energy = decay ** 2
            # Backward cumulative sum (Schroeder integral)
            schroeder = np.cumsum(energy[::-1])[::-1]
            schroeder = schroeder / (schroeder[0] + 1e-9)
            schroeder_db = 10 * np.log10(schroeder + 1e-9)

            # Find -5 dB and -25 dB crossing (T20 → extrapolate to T60)
            t = np.arange(len(schroeder_db)) / sr
            try:
                idx_5  = np.where(schroeder_db <= -5)[0][0]
                idx_25 = np.where(schroeder_db <= -25)[0][0]
                t20    = t[idx_25] - t[idx_5]
                rt60   = t20 * 3.0   # extrapolate T20 → T60
                return float(np.clip(rt60, 0.0, 5.0))
            except IndexError:
                return 0.01   # decay too fast → no reverb

        except Exception:
            return None

    def _spectral_flatness(self, audio: np.ndarray, sr: int) -> Optional[float]:
        """Compute mean spectral flatness (Wiener entropy). Near 1 = white noise, near 0 = tonal."""
        try:
            if LIBROSA_OK:
                sf = librosa.feature.spectral_flatness(y=audio)
                return float(np.mean(sf))
            # Fallback via FFT
            fft_mag = np.abs(np.fft.rfft(audio[:sr]))  # 1 second
            geo_mean = np.exp(np.mean(np.log(fft_mag + 1e-9)))
            arith_mean = np.mean(fft_mag) + 1e-9
            return float(geo_mean / arith_mean)
        except Exception:
            return None

    def _zcr_variance(self, audio: np.ndarray, sr: int) -> Optional[float]:
        """Compute variance of zero-crossing rate. Synthetic audio has unnaturally low variance."""
        try:
            frame_len = int(sr * 0.025)
            zcrs = []
            for i in range(0, len(audio) - frame_len, frame_len):
                frame = audio[i:i + frame_len]
                zcr   = np.mean(np.abs(np.diff(np.sign(frame)))) / 2
                zcrs.append(zcr)
            if not zcrs:
                return None
            return float(np.var(zcrs))
        except Exception:
            return None


# ── Singleton ─────────────────────────────────────────────────────────────────
acoustic_analyzer = AcousticAnalyzer()
