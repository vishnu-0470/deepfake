"""
tests/conftest.py — shared pytest fixtures
"""
import pytest
import numpy as np
import cv2
import asyncio
from unittest.mock import AsyncMock, patch


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def real_frames():
    frames = []
    for i in range(90):
        f = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.ellipse(f, (320, 240), (90, 110), 0, 0, 360, (200, 155, 110), -1)
        pulse = int(8 * np.sin(2 * np.pi * 1.2 * i / 30))
        f[210:250, 280:370] = np.clip(f[210:250, 280:370].astype(int) + [pulse, -pulse//2, 0], 0, 255)
        frames.append(f)
    return frames


@pytest.fixture
def fake_frames():
    base = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.ellipse(base, (320, 240), (90, 110), 0, 0, 360, (200, 155, 110), -1)
    return [base.copy() for _ in range(90)]


@pytest.fixture
def mock_session():
    return {
        "session_id":    "test-session-001",
        "session_token": "test-token-abc",
        "applicant_name": "Test User",
        "id_type":       "AADHAAR",
        "illum_colors":  ["#FF0000", "#0000FF", "#00FF00", "#FFFFFF"],
        "liveness":      "Look LEFT",
        "status":        "ACTIVE",
        "doc_path":      None,
        "audio_paths":   [],
        "frames":        [],
        "result":        None,
    }
