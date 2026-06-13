# DeepShield KYC

**Next-Generation Deepfake Defense for Web-Based Video KYC**

A multi-layer, industry-grade AI system that detects deepfakes, synthetic audio,
virtual cameras, and identity fraud in real-time during bank KYC video calls.

---

## Architecture Overview

```
Browser (WebCam + Mic)
        │
        ▼ WebSocket (JPEG frames) + REST (audio chunks, document)
   FastAPI Backend
        │
        ├── Layer 1: EfficientNet-B4 Deepfake Classifier (FF++ trained)
        ├── Layer 2: rPPG Biological Signal (CHROM method)
        ├── Layer 3: Acoustic Noise Profiling (RT60 + SNR)
        ├── Layer 4: Screen Illumination Challenge
        ├── Layer 5: Document OCR + ArcFace Face Match
        └── Layer 6: Hardware Camera Auth (C binary)
                │
                ▼
        Weighted Ensemble Scorer
                │
                ▼
        Risk Score 0–100 → APPROVED / REVIEW / BLOCKED
```

---

## Detection Layers

| Layer | Method | Catches |
|---|---|---|
| Deepfake Classifier | EfficientNet-B4 + 2D FFT | Face-swap, GAN-generated faces |
| rPPG | CHROM algorithm (heart rate) | Static deepfakes, 2D video injection |
| Acoustic Profiling | RT60 + SNR + ZCR | AI voice cloning, audio injection |
| Illumination Challenge | Screen flash + face color correlation | Pre-rendered deepfakes, live face-swap |
| Face Match | ArcFace cosine similarity (InsightFace) | Identity fraud, stolen documents |
| Hardware Auth | C syscall → `/sys/class/video4linux/` | OBS Virtual Camera, DeepFaceLive |

---

## Quick Start

### Prerequisites
- Python 3.11+
- GCC (for C binary)
- FFmpeg
- Tesseract OCR
- Redis (optional — falls back to in-memory)
- CUDA GPU (optional — CPU works but slower)

### 1. Setup

```bash
git clone https://github.com/your-team/deepshield-kyc
cd deepshield-kyc
make setup
cp .env.example .env
# Edit .env — set your SECRET_KEY
```

### 2. Run (Development)

```bash
make dev-redis     # start Redis in Docker
make dev           # start FastAPI with hot reload
```

Open `http://localhost:8000` in your browser.

### 3. Run (Docker — Production)

```bash
make docker-up
```

Open `http://localhost` — nginx proxies to the app.

---

## Project Structure

```
deepshield/
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Centralised settings
│   ├── models/
│   │   └── schemas.py             # Pydantic data models
│   ├── routers/
│   │   ├── kyc.py                 # REST endpoints
│   │   └── ws_router.py           # WebSocket endpoint
│   ├── detectors/
│   │   ├── deepfake_classifier.py # EfficientNet-B4 + FFT
│   │   ├── rppg_detector.py       # rPPG (CHROM method)
│   │   ├── acoustic_analyzer.py   # RT60 + SNR profiling
│   │   ├── illumination_challenge.py # Screen flash challenge
│   │   ├── document_ocr.py        # OCR + ArcFace face match
│   │   └── hardware_checker.py    # Virtual camera detection
│   ├── pipeline/
│   │   ├── orchestrator.py        # Parallel layer orchestration
│   │   └── session_store.py       # Redis-backed session storage
│   └── utils/
│       └── scoring.py             # Weighted ensemble risk scorer
├── frontend/
│   ├── index.html                 # KYC UI
│   └── static/
│       ├── style.css              # Dark theme CSS
│       └── app.js                 # WebSocket client + challenge logic
├── hardware/
│   └── camera_auth.c              # C binary for hardware camera check
├── tests/
│   └── test_pipeline.py           # Full test suite
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── Makefile
├── requirements.txt
└── .env.example
```

---

## Model Checkpoint

The deepfake classifier requires a fine-tuned EfficientNet-B4 checkpoint.

**Option A — Use pretrained weights (hackathon mode)**
The system works with ImageNet pretrained weights but with reduced accuracy.
Set `CLASSIFIER_CHECKPOINT` to a non-existent path and the system falls back automatically.

**Option B — Fine-tuned on FaceForensics++**
1. Download FF++ dataset: https://github.com/ondyari/FaceForensics
2. Fine-tune: `python scripts/train_classifier.py --data /path/to/ff++`
3. Place checkpoint at `models/deepfake_effnetb4_ff++.pth`

**Option C — Use a public HuggingFace checkpoint**
Search `huggingface.co` for `efficientnet deepfake detection` — several FF++ checkpoints are available.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/kyc/session` | Create KYC session |
| POST | `/api/kyc/upload-doc` | Upload ID document |
| POST | `/api/kyc/upload-audio` | Upload audio chunk |
| GET | `/api/kyc/result/{id}` | Fetch analysis result |
| WS | `/ws/kyc/{session_id}` | Live video stream |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |
| GET | `/metrics` | Prometheus metrics |

---

## Risk Score Interpretation

| Score | Verdict | Action |
|---|---|---|
| 0 – 39 | APPROVED | KYC passed — proceed to account opening |
| 40 – 69 | REVIEW | Flag for manual officer review |
| 70 – 100 | BLOCKED | KYC rejected — fraud alert raised |

---

## Running Tests

```bash
make test           # full suite
make test-fast      # stop on first failure
make test-cov       # with coverage report
```

---

## Hackathon Demo Script

1. Open `http://localhost:8000`
2. **Scenario 1 (Real customer)**: Allow camera, complete liveness, submit → APPROVED
3. **Scenario 2 (Virtual camera)**: Start OBS Virtual Camera, repeat → BLOCKED
4. **Scenario 3 (ID mismatch)**: Upload a different person's ID → REVIEW/BLOCKED
5. Show the live risk score updating layer by layer in real time

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, uvicorn |
| ML Framework | PyTorch 2.3, timm (EfficientNet-B4) |
| Face Detection | MediaPipe, InsightFace (ArcFace) |
| Signal Processing | NumPy, SciPy, librosa |
| Audio Processing | librosa, ffmpeg |
| OCR | Tesseract, EasyOCR |
| Hardware Auth | C (Linux syscall / IOKit) |
| Session Store | Redis |
| Async | asyncio, FastAPI WebSockets |
| Monitoring | Prometheus + Grafana |
| Deployment | Docker, Docker Compose, nginx |
