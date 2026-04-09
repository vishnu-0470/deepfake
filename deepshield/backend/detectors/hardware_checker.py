"""
backend/detectors/hardware_checker.py
─────────────────────────────────────────────────────────
DeepShield KYC  –  Hardware Camera Authentication (Python)

Calls the compiled C binary `camera_auth` and parses JSON.
Falls back to a pure-Python heuristic if the binary is not
available (e.g., during development on macOS).
─────────────────────────────────────────────────────────
"""

import asyncio
import json
import subprocess
import time
from pathlib import Path
from loguru import logger

from backend.config import get_settings
from backend.models.schemas import HardwareAuthResult, DetectionLabel

settings = get_settings()


class HardwareChecker:
    """Validates camera device is a physical hardware webcam."""

    async def analyze(self) -> HardwareAuthResult:
        t0 = time.perf_counter()

        if not settings.HW_AUTH_ENABLED:
            return HardwareAuthResult(
                label=DetectionLabel.UNKNOWN,
                confidence=0.5,
                risk_contribution=0.0,
                detail="Hardware check disabled in config",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._run_check
            )
        except Exception as e:
            logger.exception(f"[HW] Error: {e}")
            result = HardwareAuthResult(
                label=DetectionLabel.UNKNOWN,
                confidence=0.5,
                risk_contribution=20.0,
                detail=f"Hardware check error: {e}",
            )

        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def _run_check(self) -> HardwareAuthResult:
        binary = Path(settings.HW_AUTH_BINARY)

        # ── Try compiled C binary ────────────────────────────────────────────
        if binary.exists():
            try:
                proc = subprocess.run(
                    [str(binary)],
                    capture_output=True, text=True, timeout=10
                )
                data = json.loads(proc.stdout.strip())
                return self._parse_binary_output(data, proc.returncode)
            except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"[HW] Binary check failed: {e} — falling back to Python")

        # ── Python fallback ───────────────────────────────────────────────────
        return self._python_fallback()

    def _parse_binary_output(self, data: dict, exit_code: int) -> HardwareAuthResult:
        is_virtual   = data.get("is_virtual", False)
        confidence   = float(data.get("confidence", 0.5))
        device_name  = data.get("device_name", "")
        vendor_id    = data.get("vendor_id", "")
        driver       = data.get("driver", "")
        reason       = data.get("reason", "")

        if is_virtual:
            label  = DetectionLabel.FAKE
            risk   = 80.0
            detail = f"Virtual camera detected: {device_name} (driver: {driver})"
        elif exit_code == 2:
            label  = DetectionLabel.UNKNOWN
            risk   = 25.0
            detail = f"Inconclusive: {reason}"
        else:
            label  = DetectionLabel.REAL
            risk   = 0.0
            detail = f"Physical camera: {device_name} (vendor: {vendor_id})"

        logger.info(f"[HW] label={label} device={device_name} virtual={is_virtual}")

        return HardwareAuthResult(
            label=label,
            confidence=round(confidence, 3),
            risk_contribution=risk,
            detail=detail,
            device_name=device_name or None,
            is_virtual=is_virtual,
            vendor_id=vendor_id or None,
        )

    def _python_fallback(self) -> HardwareAuthResult:
        """
        Pure Python fallback: check /proc/modules and /dev/video* on Linux,
        or check running processes for known virtual camera apps on any OS.
        """
        import platform
        import os

        os_name = platform.system()

        # Check for virtual camera processes
        virtual_process_names = [
            "obs", "obs-studio", "manycam", "xsplit", "snapcamera",
            "droidcam", "epoccam", "ivcam", "deepfacelive"
        ]
        try:
            proc_list = subprocess.run(
                ["ps", "aux"] if os_name != "Windows" else ["tasklist"],
                capture_output=True, text=True, timeout=5
            )
            output_lower = proc_list.stdout.lower()
            for vp in virtual_process_names:
                if vp in output_lower:
                    return HardwareAuthResult(
                        label=DetectionLabel.FAKE,
                        confidence=0.82,
                        risk_contribution=75.0,
                        detail=f"Virtual camera application running: {vp}",
                        is_virtual=True,
                    )
        except Exception:
            pass

        # Linux: check loaded kernel modules
        if os_name == "Linux":
            try:
                modules_path = "/proc/modules"
                if os.path.exists(modules_path):
                    with open(modules_path) as f:
                        modules = f.read().lower()
                    if "v4l2loopback" in modules:
                        return HardwareAuthResult(
                            label=DetectionLabel.FAKE,
                            confidence=0.88,
                            risk_contribution=78.0,
                            detail="v4l2loopback kernel module active (virtual camera driver)",
                            is_virtual=True,
                        )
            except Exception:
                pass

        # Default: assume physical (low confidence)
        return HardwareAuthResult(
            label=DetectionLabel.REAL,
            confidence=0.55,
            risk_contribution=5.0,
            detail="No virtual camera processes detected (Python fallback)",
            is_virtual=False,
        )


# ── Singleton ──────────────────────────────────────────────────────────────────
hardware_checker = HardwareChecker()
