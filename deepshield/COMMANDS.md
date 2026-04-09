# DeepShield KYC — Complete Run Commands Reference

Every command you need, in order.

---

## STEP 0 — Prerequisites (install once)

### Ubuntu / Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    gcc g++ make \
    ffmpeg \
    tesseract-ocr tesseract-ocr-eng \
    libgl1-mesa-glx libglib2.0-0 \
    git curl docker.io
```

### macOS (with Homebrew)
```bash
# Install Homebrew if missing
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install python@3.11 ffmpeg tesseract gcc git
```

### Windows (WSL2 recommended)
```bash
# Run Ubuntu via WSL2, then follow Ubuntu steps above
wsl --install -d Ubuntu-22.04
```

---

## STEP 1 — Get the project

```bash
# Unzip the downloaded file
unzip deepshield_kyc_v2.zip
cd deepshield

# OR clone from your repo
# git clone https://github.com/your-team/deepshield && cd deepshield
```

---

## STEP 2 — One-shot automated setup

```bash
chmod +x setup.sh
./setup.sh
```

This script automatically:
- Creates a Python virtual environment
- Installs all Python packages
- Compiles the C hardware auth binary
- Creates `.env` with a random secret key
- Creates `uploads/`, `models/`, `logs/` directories
- Generates demo test assets

---

## STEP 3 — (If setup.sh fails) Manual install

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate           # Linux/macOS
# .venv\Scripts\activate            # Windows

# Install dependencies
pip install --upgrade pip wheel
pip install -r requirements.txt

# Compile C binary (Linux/macOS)
gcc -O2 -o hardware/camera_auth hardware/camera_auth.c
chmod +x hardware/camera_auth

# Create directories
mkdir -p uploads models logs

# Set up env
cp .env.example .env
```

---

## STEP 4 — Edit your .env file

```bash
# Open .env in any editor
nano .env
# or
code .env

# The ONLY required change:
SECRET_KEY=any_long_random_string_here_at_least_32_chars

# Optional tweaks:
DEBUG=true          # Enable hot reload and debug logs
PORT=8000           # Change port if 8000 is taken
HW_AUTH_ENABLED=true
```

---

## STEP 5 — Run the server

### Option A — Fastest (no Docker needed)
```bash
source .venv/bin/activate
chmod +x run.sh
./run.sh
```

Or manually:
```bash
source .venv/bin/activate
python3 -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Option B — With Redis (recommended for production demo)
```bash
# Terminal 1: Start Redis
docker run -d --rm --name ds_redis -p 6379:6379 redis:7-alpine

# Terminal 2: Start server
source .venv/bin/activate
./run.sh
```

### Option C — Full Docker Compose (everything)
```bash
# Build and start all services
docker-compose up --build

# Or in background
docker-compose up --build -d

# View logs
docker-compose logs -f app

# Stop everything
docker-compose down
```

---

## STEP 6 — Open in browser

```
http://localhost:8000          → KYC UI
http://localhost:8000/docs     → Swagger API docs
http://localhost:8000/health   → Health check
http://localhost:8000/metrics  → Prometheus metrics
```

---

## STEP 7 — Generate demo assets (for judges)

```bash
source .venv/bin/activate
python3 scripts/demo_test.py

# Creates:
#   demo_assets/demo_real.mp4           → simulated real face video
#   demo_assets/demo_fake.mp4           → simulated deepfake video
#   demo_assets/test_id_aadhaar.jpg     → test ID document
```

---

## STEP 8 — (Optional) Train the deepfake classifier

```bash
source .venv/bin/activate

# Download FaceForensics++ dataset first, then:
python3 scripts/train_classifier.py \
    --data  /path/to/ff++/data \
    --out   models/deepfake_effnetb4_ff++.pth \
    --epochs 10 \
    --batch  16

# For quick hackathon training (2 hours on GPU):
python3 scripts/train_classifier.py \
    --data  /path/to/ff++/data \
    --out   models/deepfake_effnetb4_ff++.pth \
    --epochs 5 \
    --batch  16 \
    --max-per-class 1000

# Alternative: download a public pretrained checkpoint
python3 scripts/download_model.py
```

---

## STEP 9 — Run tests

```bash
source .venv/bin/activate

# Full test suite
python3 -m pytest tests/ -v

# Quick smoke test (stop on first failure)
python3 -m pytest tests/ -x -q

# With coverage report
python3 -m pytest tests/ --cov=backend --cov-report=html
open htmlcov/index.html
```

---

## STEP 10 — Hardware binary

```bash
# Compile (Linux)
gcc -O2 -o hardware/camera_auth hardware/camera_auth.c
chmod +x hardware/camera_auth

# Compile (macOS — requires IOKit)
clang -O2 -framework IOKit -framework CoreFoundation \
      -o hardware/camera_auth hardware/camera_auth.c

# Test the binary
./hardware/camera_auth
# Expected output: {"is_virtual": false, "device": "/dev/video0", ...}
```

---

## Troubleshooting

### Port already in use
```bash
kill $(lsof -t -i:8000)
# or
./run.sh --port 8001
```

### Camera permission denied
```bash
# Linux: add yourself to video group
sudo usermod -aG video $USER
# Then log out and back in

# macOS: grant camera permission in System Settings → Privacy → Camera
```

### mediapipe install fails
```bash
pip install mediapipe --no-build-isolation
# or skip it — rPPG will use fallback
```

### torch takes too long to install
```bash
# CPU-only (faster download, no GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### insightface build error
```bash
pip install insightface --no-build-isolation
# or skip — face matching will use OpenCV Haar cascade fallback
```

### pyaudio fails on Linux
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

### pyaudio fails on macOS
```bash
brew install portaudio
pip install pyaudio
```

### Redis connection refused
```bash
# Start Redis
docker run -d --rm -p 6379:6379 redis:7-alpine

# Or install Redis natively:
# Ubuntu:  sudo apt install redis-server && sudo systemctl start redis
# macOS:   brew install redis && brew services start redis
```

### tesseract not found
```bash
# Ubuntu
sudo apt install tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract

# Then update .env:
TESSERACT_CMD=/usr/bin/tesseract        # Linux
TESSERACT_CMD=/opt/homebrew/bin/tesseract  # macOS Apple Silicon
```

---

## Quick reference

| Command | What it does |
|---|---|
| `./setup.sh` | Full automated setup (run first) |
| `./run.sh` | Start server with auto-Redis |
| `./run.sh --prod` | Production mode (no hot reload) |
| `./run.sh --port 8001` | Run on custom port |
| `docker-compose up --build` | Full stack with Docker |
| `python3 scripts/demo_test.py` | Generate demo assets |
| `python3 -m pytest tests/ -v` | Run all tests |
| `docker-compose down` | Stop all services |
| `make lint` | Run linter |
| `make clean` | Remove temp files |
