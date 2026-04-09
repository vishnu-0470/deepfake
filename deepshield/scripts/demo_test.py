"""
scripts/demo_test.py
─────────────────────────────────────────────────────────
DeepShield KYC  –  Demo Test Video Generator

Creates synthetic test cases for demonstrating the system
to judges WITHOUT needing a real webcam:

  1. demo_real.mp4   – simulated real face (with pulse, reverb)
  2. demo_fake.mp4   – simulated deepfake (static, no pulse)

Usage:
  python scripts/demo_test.py
  # Outputs to: demo_assets/

Then open the browser and use these pre-recorded videos
instead of your webcam to guarantee a smooth demo.
─────────────────────────────────────────────────────────
"""

import os
import subprocess
from pathlib import Path

import numpy as np

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    print("cv2 not found. Install: pip install opencv-python")

try:
    import soundfile as sf
    SF_OK = True
except ImportError:
    SF_OK = False


OUT_DIR  = Path("demo_assets")
DURATION = 10        # seconds
FPS      = 30
FRAMES   = DURATION * FPS
SR       = 16000
WIDTH    = 640
HEIGHT   = 480


def _face_base(frame_idx: int, real: bool) -> np.ndarray:
    """Draw a simple face on a dark background."""
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Background gradient
    for y in range(HEIGHT):
        v = int(20 + y * 0.04)
        img[y, :] = [v, v+4, v+8]

    cx, cy = WIDTH // 2, HEIGHT // 2

    # Skin base
    cv2.ellipse(img, (cx, cy), (90, 115), 0, 0, 360, (185, 140, 100), -1)

    # Eyes
    for ex in [cx-28, cx+28]:
        cv2.ellipse(img, (ex, cy-30), (12, 8), 0, 0, 360, (30, 25, 20), -1)
        cv2.circle(img, (ex, cy-30), 4, (80, 60, 40), -1)
        cv2.circle(img, (ex-2, cy-32), 2, (220, 220, 220), -1)

    # Nose
    cv2.line(img, (cx, cy-10), (cx, cy+20), (160, 120, 85), 1)
    cv2.ellipse(img, (cx, cy+20), (12, 5), 0, 0, 180, (160, 120, 85), 1)

    # Mouth
    pts = np.array([[cx-25, cy+45], [cx, cy+55], [cx+25, cy+45]], np.int32)
    cv2.polylines(img, [pts], False, (130, 80, 60), 2)

    if real:
        # Simulate rPPG: subtle skin color oscillation (heart rate ~72 bpm = 1.2 Hz)
        t      = frame_idx / FPS
        pulse  = int(8 * np.sin(2 * np.pi * 1.2 * t))
        mask   = np.zeros_like(img)
        cv2.ellipse(mask, (cx, cy-40), (60, 30), 0, 0, 360, (1, 1, 1), -1)
        img = np.clip(img.astype(int) + mask * pulse, 0, 255).astype(np.uint8)

        # Random micro head movements
        if frame_idx % 15 == 0:
            offset = np.random.randint(-2, 3, 2)
            M      = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
            img    = cv2.warpAffine(img, M, (WIDTH, HEIGHT))
    else:
        # Deepfake: perfectly static (no motion, no pulse variation)
        # Add subtle GAN-like grid artifact in high-frequency
        grid = np.zeros_like(img)
        grid[::8, :] = 3
        grid[:, ::8] = 3
        img = np.clip(img.astype(int) + grid, 0, 255).astype(np.uint8)

    # Timestamp overlay
    cv2.putText(img, f"{'REAL' if real else 'DEEPFAKE'} DEMO  t={frame_idx/FPS:.1f}s",
                (12, HEIGHT-14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (0, 255, 140) if real else (255, 59, 92), 1, cv2.LINE_AA)

    return img


def make_audio(real: bool) -> np.ndarray:
    """Generate audio: real = noise + reverb, fake = clean pure tone."""
    t = np.linspace(0, DURATION, SR * DURATION)

    # Speech-like carrier
    speech = 0.3 * np.sin(2 * np.pi * 180 * t) + 0.1 * np.sin(2 * np.pi * 360 * t)

    if real:
        # Add room noise + exponential reverb
        noise  = 0.04 * np.random.randn(len(t))
        reverb = np.convolve(speech, np.exp(-np.linspace(0, 8, SR // 5)), mode="same") * 0.05
        audio  = speech + noise + reverb
    else:
        # Unnaturally clean: pure tone, no noise
        audio = speech * 0.9

    return audio.astype(np.float32)


def make_video(name: str, real: bool):
    if not CV2_OK:
        return

    out_path = OUT_DIR / f"{name}.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    vw       = cv2.VideoWriter(str(out_path), fourcc, FPS, (WIDTH, HEIGHT))

    print(f"Generating {'real' if real else 'fake'} demo video: {out_path}")
    for i in range(FRAMES):
        frame = _face_base(i, real)
        vw.write(frame)
        if i % (FPS * 2) == 0:
            print(f"  {i // FPS}s / {DURATION}s", end="\r")

    vw.release()
    print(f"\n  ✓ Video saved: {out_path}")

    # Save audio separately
    if SF_OK:
        audio     = make_audio(real)
        audio_path = OUT_DIR / f"{name}_audio.wav"
        sf.write(str(audio_path), audio, SR)
        print(f"  ✓ Audio saved: {audio_path}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("=" * 55)
    print("  DeepShield KYC – Demo Asset Generator")
    print("=" * 55)
    make_video("demo_real", real=True)
    make_video("demo_fake", real=False)

    # Create a simple Aadhaar-like test ID image
    if CV2_OK:
        id_img = np.full((300, 500, 3), 245, dtype=np.uint8)
        cv2.rectangle(id_img, (0, 0), (500, 300), (0, 120, 60), 4)
        cv2.putText(id_img, "GOVERNMENT OF INDIA", (80, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 50), 2)
        cv2.putText(id_img, "AADHAAR  CARD",       (140, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 40), 1)
        cv2.line(id_img, (20, 90), (480, 90), (0, 120, 60), 1)
        cv2.putText(id_img, "Name:  Priya Sharma",  (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
        cv2.putText(id_img, "DOB:   15/08/1995",    (30, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
        cv2.putText(id_img, "Aadhaar: 4821 XXXX XXXX", (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
        # Face placeholder
        cv2.rectangle(id_img, (360, 110), (470, 220), (200, 200, 200), -1)
        cv2.ellipse(id_img, (415, 155), (30, 38), 0, 0, 360, (185, 140, 100), -1)
        id_path = OUT_DIR / "test_id_aadhaar.jpg"
        cv2.imwrite(str(id_path), id_img)
        print(f"  ✓ Test ID image: {id_path}")

    print(f"\nAll demo assets saved to: {OUT_DIR}/")
    print("\nUsage in demo:")
    print("  Upload test_id_aadhaar.jpg as your document")
    print("  For live demo, use your actual webcam")


if __name__ == "__main__":
    main()
