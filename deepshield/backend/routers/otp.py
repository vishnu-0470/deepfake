"""
backend/routers/otp.py
─────────────────────────────────────────────────────────
DeepShield KYC  –  OTP via Twilio SMS
  POST /api/otp/send    → send OTP to phone number
  POST /api/otp/verify  → verify OTP entered by user
─────────────────────────────────────────────────────────
"""

import random
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

from backend.config import get_settings
from backend.pipeline.session_store import session_store

router = APIRouter()
settings = get_settings()

OTP_EXPIRY_SECONDS = 300  # 5 minutes


class SendOTPRequest(BaseModel):
    phone: str
    session_id: str


class VerifyOTPRequest(BaseModel):
    phone: str
    otp: str
    session_id: str


def _send_sms(phone: str, otp: str) -> bool:
    """Send OTP via Twilio. Falls back to console log if not configured."""
    if not settings.TWILIO_ACCOUNT_SID or settings.TWILIO_ACCOUNT_SID == "your_twilio_account_sid":
        # Dev mode: just log the OTP
        logger.info(f"[OTP] DEV MODE — OTP for {phone}: {otp}")
        return True
    try:
        from twilio.rest import Client
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=f"Your DeepShield KYC verification code is: {otp}. Valid for 5 minutes.",
            from_=settings.TWILIO_PHONE_NUMBER,
            to=phone,
        )
        logger.info(f"[OTP] SMS sent to {phone}")
        return True
    except Exception as e:
        logger.error(f"[OTP] Twilio error: {e}")
        return False


@router.post("/send")
async def send_otp(req: SendOTPRequest):
    session = await session_store.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Generate 6-digit OTP
    otp = str(random.randint(100000, 999999))
    expiry = time.time() + OTP_EXPIRY_SECONDS

    # Store OTP in session
    session["otp"] = otp
    session["otp_expiry"] = expiry
    session["otp_phone"] = req.phone
    session["otp_verified"] = False
    await session_store.set(req.session_id, session)

    sent = _send_sms(req.phone, otp)
    if not sent:
        raise HTTPException(status_code=500, detail="Failed to send OTP. Check Twilio config.")

    logger.info(f"[OTP] Sent to {req.phone} for session {req.session_id}")

    # In dev mode return OTP in response so frontend can show it
    dev_mode = not settings.TWILIO_ACCOUNT_SID or settings.TWILIO_ACCOUNT_SID == "your_twilio_account_sid"
    return JSONResponse({
        "ok": True,
        "message": f"OTP sent to {req.phone}",
        "dev_otp": otp if dev_mode else None,
        "dev_mode": dev_mode,
    })


@router.post("/verify")
async def verify_otp(req: VerifyOTPRequest):
    session = await session_store.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    stored_otp = session.get("otp")
    expiry = session.get("otp_expiry", 0)

    if not stored_otp:
        raise HTTPException(status_code=400, detail="No OTP sent for this session")

    if time.time() > expiry:
        raise HTTPException(status_code=400, detail="OTP expired. Please request a new one.")

    if req.otp.strip() != stored_otp:
        raise HTTPException(status_code=400, detail="Incorrect OTP. Please try again.")

    # Mark verified
    session["otp_verified"] = True
    await session_store.set(req.session_id, session)

    logger.info(f"[OTP] Verified for session {req.session_id}")
    return JSONResponse({"ok": True, "message": "Phone number verified successfully"})
