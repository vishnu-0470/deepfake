# DeepShield KYC — Product Overview

## Purpose & Value Proposition
DeepShield KYC is a next-generation, multi-layer AI system that detects deepfakes, synthetic audio, virtual cameras, and identity fraud in real-time during bank/financial KYC (Know Your Customer) video calls. It provides a risk score (0–100) that drives automated APPROVED / REVIEW / BLOCKED verdicts.

## Key Features & Capabilities

### 6-Layer Detection Pipeline
| Layer | Method | Catches |
|---|---|---|
| Deepfake Classifier | EfficientNet-B4 + 2D FFT | Face-swap, GAN-generated faces |
| rPPG Biological Signal | CHROM algorithm (heart rate) | Static deepfakes, 2D video injection |
| Acoustic Profiling | RT60 + SNR + ZCR | AI voice cloning, audio injection |
| Illumination Challenge | Screen flash + face color correlation | Pre-rendered deepfakes, live face-swap |
| Document OCR + Face Match | ArcFace cosine similarity (InsightFace) | Identity fraud, stolen documents |
| Hardware Camera Auth | C syscall → /sys/class/video4linux/ | OBS Virtual Camera, DeepFaceLive |

### Risk Scoring
- Score 0–39 → **APPROVED** (KYC passed)
- Score 40–69 → **REVIEW** (manual officer review)
- Score 70–100 → **BLOCKED** (fraud alert raised)

### API Surface
| Method | Endpoint | Description |
|---|---|---|
| POST | /api/kyc/session | Create KYC session |
| POST | /api/kyc/upload-doc | Upload ID document |
| POST | /api/kyc/upload-audio | Upload audio chunk |
| GET | /api/kyc/result/{id} | Fetch analysis result |
| WS | /ws/kyc/{session_id} | Live video stream |
| POST | /api/otp/send | Send OTP via Twilio |
| GET | /health | Health check |
| GET | /metrics | Prometheus metrics |

### Real-Time Processing
- WebSocket-based live video frame analysis (JPEG frames)
- Parallel layer orchestration via asyncio
- Redis-backed session storage with in-memory fallback
- Live risk score updates layer-by-layer in the browser UI

## Target Users & Use Cases
- **Primary**: Banks and financial institutions running video KYC onboarding
- **Use Case 1**: Real customer verification — camera + liveness + document match → APPROVED
- **Use Case 2**: Virtual camera / OBS injection detection → BLOCKED
- **Use Case 3**: ID document mismatch / stolen identity → REVIEW / BLOCKED
- **Secondary**: Any platform requiring liveness verification and anti-spoofing
