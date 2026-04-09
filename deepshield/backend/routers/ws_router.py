"""
backend/routers/ws_router.py
─────────────────────────────────────────────────────────
DeepShield KYC  –  WebSocket Endpoint
  WS /ws/kyc/{session_id}

Message flow:
  Client → Server : raw JPEG bytes  (video frames)
  Client → Server : JSON { type: "CHALLENGE_COMPLETE" }
  Client → Server : JSON { type: "SUBMIT" }
  Server → Client : JSON { type: "LAYER_UPDATE", payload: {...} }
  Server → Client : JSON { type: "CHALLENGE_READY", payload: {...} }
  Server → Client : JSON { type: "ANALYSIS_COMPLETE", payload: {...} }
─────────────────────────────────────────────────────────
"""

import asyncio
import json
from datetime import datetime

import numpy as np
import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from loguru import logger

from backend.pipeline.session_store import session_store
from backend.pipeline.orchestrator import pipeline
from backend.models.schemas import WSMessage, WSMessageType

router = APIRouter()

MAX_FRAMES_STORED = 300   # ~60 sec @ 5 fps


@router.websocket("/kyc/{session_id}")
async def kyc_websocket(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(...),
):
    await websocket.accept()
    logger.info(f"[WS] Client connected: {session_id}")

    session = await session_store.get(session_id)
    if not session or session.get("session_token") != token:
        await websocket.close(code=4001, reason="Invalid session or token")
        return

    frames_buffer: list[np.ndarray] = []
    challenge_done = asyncio.Event()

    try:
        # Trigger illumination challenge shortly after connection
        asyncio.get_event_loop().call_later(
            5.0,
            lambda: asyncio.ensure_future(
                _send_json(websocket, {
                    "type": WSMessageType.CHALLENGE_READY,
                    "payload": {"colors": session["illum_colors"]},
                })
            ),
        )

        while True:
            raw = await websocket.receive()

            # ── Binary frame ─────────────────────────────────────────────────
            if "bytes" in raw and raw["bytes"]:
                frame = _decode_frame(raw["bytes"])
                if frame is not None:
                    frames_buffer.append(frame)
                    # Keep ring buffer
                    if len(frames_buffer) > MAX_FRAMES_STORED:
                        frames_buffer.pop(0)

                    # Stream live per-frame updates every 30 frames (~6 sec)
                    if len(frames_buffer) % 30 == 0:
                        await _run_live_updates(websocket, session_id, frames_buffer[-30:])

            # ── JSON control messages ─────────────────────────────────────────
            elif "text" in raw and raw["text"]:
                msg = json.loads(raw["text"])
                msg_type = msg.get("type", "")

                if msg_type == "CHALLENGE_COMPLETE":
                    challenge_done.set()
                    logger.info(f"[WS] Challenge complete for {session_id}")

                elif msg_type == "SUBMIT":
                    logger.info(f"[WS] Submit triggered for {session_id} — {len(frames_buffer)} frames")

                    # Store frames + illum start frame to session
                    session["frames"] = frames_buffer
                    session["illum_start_frame"] = max(0, len(frames_buffer) - 60)
                    await session_store.set(session_id, session)

                    # Wait briefly for any in-flight audio uploads to land
                    await asyncio.sleep(3.5)

                    # Re-fetch session to get latest audio_paths
                    session = await session_store.get(session_id)

                    # Run full analysis pipeline
                    result = await pipeline.run(
                        session_id=session_id,
                        frames=frames_buffer,
                        on_layer_update=lambda update: asyncio.ensure_future(
                            _send_json(websocket, {
                                "type": WSMessageType.LAYER_UPDATE,
                                "payload": update,
                            })
                        ),
                    )

                    # Store result
                    session["result"] = result.model_dump()
                    await session_store.set(session_id, session)

                    # Send final result
                    await _send_json(websocket, {
                        "type": WSMessageType.ANALYSIS_COMPLETE,
                        "payload": result.model_dump(mode="json"),
                    })
                    break

    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnected: {session_id}")
    except Exception as e:
        logger.exception(f"[WS] Unexpected error: {e}")
        await _send_json(websocket, {
            "type": WSMessageType.ERROR,
            "payload": {"detail": str(e)},
        })
    finally:
        await websocket.close()


async def _run_live_updates(websocket: WebSocket, session_id: str, frames: list):
    """Send incremental layer results during recording (before final submit)."""
    try:
        # Quick rPPG check (just signal presence)
        from backend.detectors.rppg_detector import rppg_detector
        partial = await rppg_detector.quick_check(frames)
        await _send_json(websocket, {
            "type": WSMessageType.LAYER_UPDATE,
            "payload": {
                "layer":  "rppg",
                "status": "running" if partial is None else ("pass" if partial else "fail"),
                "detail": "Measuring pulse..." if partial is None else (
                    "Signal detected" if partial else "No biological signal"
                ),
            },
        })
    except Exception:
        pass


async def _send_json(websocket: WebSocket, data: dict):
    try:
        await websocket.send_text(json.dumps(data, default=str))
    except Exception:
        pass


def _decode_frame(raw_bytes: bytes) -> np.ndarray | None:
    """Decode JPEG bytes into an OpenCV BGR frame."""
    try:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None
