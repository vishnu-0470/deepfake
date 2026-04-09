#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  DeepShield KYC  –  setup.sh
#  One-shot full setup script. Run this FIRST on a fresh system.
#
#  Usage:
#    chmod +x setup.sh && ./setup.sh
#
#  Works on: Ubuntu 20.04+, macOS 12+
# ═══════════════════════════════════════════════════════════

set -e   # exit on any error

CYAN="\033[0;36m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

info()    { echo -e "${CYAN}[INFO]${RESET} $1"; }
success() { echo -e "${GREEN}[OK]${RESET}   $1"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $1"; }
error()   { echo -e "${RED}[ERR]${RESET}  $1"; }

echo ""
echo "  ██████╗ ███████╗███████╗██████╗ ███████╗██╗  ██╗██╗███████╗██╗     ██████╗ "
echo "  ██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██║  ██║██║██╔════╝██║     ██╔══██╗"
echo "  ██║  ██║█████╗  █████╗  ██████╔╝███████╗███████║██║█████╗  ██║     ██║  ██║"
echo "  ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ╚════██║██╔══██║██║██╔══╝  ██║     ██║  ██║"
echo "  ██████╔╝███████╗███████╗██║     ███████║██║  ██║██║███████╗███████╗██████╔╝"
echo "  ╚═════╝ ╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═════╝ "
echo "  KYC Verification System  ·  Full Setup"
echo ""

# ── 1. Check Python ────────────────────────────────────────────────────────
info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    error "Python 3.10+ required. Found: $PYTHON_VERSION"
    error "Install from https://www.python.org/downloads/"
    exit 1
fi
success "Python $PYTHON_VERSION"

# ── 2. Check pip ───────────────────────────────────────────────────────────
info "Checking pip..."
python3 -m pip --version > /dev/null 2>&1 || {
    warn "pip not found, installing..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
    rm get-pip.py
}
success "pip OK"

# ── 3. Create virtual environment ─────────────────────────────────────────
info "Creating virtual environment (.venv)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    success "Virtual environment created"
else
    warn ".venv already exists, skipping"
fi

# Activate
source .venv/bin/activate
success "Virtual environment activated"

# ── 4. Upgrade pip ─────────────────────────────────────────────────────────
info "Upgrading pip, wheel, setuptools..."
pip install --upgrade pip wheel setuptools --quiet
success "pip upgraded"

# ── 5. System dependencies ─────────────────────────────────────────────────
OS=$(uname -s)
info "Detected OS: $OS"

if [ "$OS" = "Linux" ]; then
    info "Installing Linux system packages..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y --no-install-recommends \
            gcc g++ make \
            ffmpeg \
            tesseract-ocr tesseract-ocr-eng \
            libgl1-mesa-glx libglib2.0-0 \
            libsm6 libxext6 libxrender-dev \
            python3-dev \
            > /dev/null 2>&1
        success "System packages installed (apt)"
    elif command -v yum &> /dev/null; then
        sudo yum install -y gcc ffmpeg tesseract > /dev/null 2>&1
        success "System packages installed (yum)"
    fi
elif [ "$OS" = "Darwin" ]; then
    info "Installing macOS packages via Homebrew..."
    if ! command -v brew &> /dev/null; then
        warn "Homebrew not found. Install from https://brew.sh"
        warn "Then re-run this script."
    else
        brew install ffmpeg tesseract > /dev/null 2>&1 || warn "Some brew packages may already be installed"
        success "Homebrew packages installed"
    fi
fi

# ── 6. Install Python dependencies ─────────────────────────────────────────
info "Installing Python dependencies (this takes 2-5 minutes)..."

# Core deps first (faster feedback)
pip install --quiet fastapi uvicorn[standard] websockets python-multipart pydantic pydantic-settings

# CV deps
pip install --quiet opencv-python Pillow numpy scipy

# ML deps (heaviest)
info "Installing PyTorch (may take a few minutes)..."
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --quiet timm

# Face detection
info "Installing face detection libs..."
pip install --quiet mediapipe || warn "mediapipe install failed — rPPG will use fallback"

# InsightFace (ArcFace) — optional, heavy
pip install --quiet insightface onnxruntime || warn "insightface not available — face match will use OpenCV"

# Audio
pip install --quiet librosa soundfile pyaudio || warn "Some audio packages failed — check ffmpeg"

# OCR
pip install --quiet pytesseract easyocr || warn "EasyOCR failed — Tesseract only"

# PDF
pip install --quiet pdf2image || warn "pdf2image failed — PDF upload won't work"

# Remaining deps from requirements.txt
pip install --quiet \
    python-jose[cryptography] passlib[bcrypt] \
    aiofiles loguru python-dotenv \
    redis anyio httpx \
    prometheus-fastapi-instrumentator \
    scikit-learn \
    || warn "Some optional packages failed"

success "Python dependencies installed"

# ── 7. Compile C binary ─────────────────────────────────────────────────────
info "Compiling camera hardware auth binary..."
if command -v gcc &> /dev/null; then
    gcc -O2 -o hardware/camera_auth hardware/camera_auth.c 2>/dev/null
    chmod +x hardware/camera_auth
    success "C binary compiled: hardware/camera_auth"
else
    warn "GCC not found — hardware auth will use Python fallback"
fi

# ── 8. Create directories ───────────────────────────────────────────────────
info "Creating required directories..."
mkdir -p uploads models logs demo_assets scripts
success "Directories ready"

# ── 9. Environment file ─────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    # Generate random secret key
    SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    sed -i.bak "s/your_secret_key_here_change_this_in_production/$SECRET/" .env
    rm -f .env.bak
    success ".env created with auto-generated secret key"
else
    warn ".env already exists — keeping existing"
fi

# ── 10. Download / prepare model ────────────────────────────────────────────
info "Preparing ML model..."
if [ ! -f "models/deepfake_effnetb4_ff++.pth" ]; then
    warn "No fine-tuned checkpoint found."
    warn "System will use ImageNet pretrained weights (~75% accuracy)."
    warn "For better accuracy: python scripts/train_classifier.py --data /path/to/ff++"
    touch models/USE_PRETRAINED
else
    success "Model checkpoint found"
fi

# ── 11. Generate demo assets ────────────────────────────────────────────────
info "Generating demo test assets..."
python3 scripts/demo_test.py 2>/dev/null && success "Demo assets ready in demo_assets/" || warn "Demo asset generation skipped"

# ── 12. Quick self-test ─────────────────────────────────────────────────────
info "Running quick import check..."
python3 -c "
import fastapi, uvicorn, cv2, numpy, torch
print('  fastapi:', fastapi.__version__)
print('  torch:  ', torch.__version__)
print('  cv2:    ', cv2.__version__)
"
success "Core imports OK"

# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}═══════════════════════════════════════════${RESET}"
echo -e "${GREEN}  Setup complete!  ${RESET}"
echo -e "${GREEN}═══════════════════════════════════════════${RESET}"
echo ""
echo "  Next steps:"
echo ""
echo "  Option A — Quick start (no Docker):"
echo "    source .venv/bin/activate"
echo "    python3 -m uvicorn backend.main:app --reload --port 8000"
echo "    open http://localhost:8000"
echo ""
echo "  Option B — Full stack with Redis:"
echo "    source .venv/bin/activate"
echo "    docker run -d -p 6379:6379 redis:7-alpine   # Redis"
echo "    python3 -m uvicorn backend.main:app --reload --port 8000"
echo "    open http://localhost:8000"
echo ""
echo "  Option C — Docker Compose (everything):"
echo "    docker-compose up --build"
echo "    open http://localhost"
echo ""
echo "  API docs:  http://localhost:8000/docs"
echo "  Health:    http://localhost:8000/health"
echo ""
