# ─────────────────────────────────────────────────────────
#  DeepShield KYC  –  Makefile
#  All dev, build, test, and deployment commands
# ─────────────────────────────────────────────────────────

.PHONY: help install dev build docker-up docker-down \
        compile-hw test lint clean logs

PYTHON     := python3
PIP        := pip3
UVICORN    := uvicorn
APP        := backend.main:app
HW_SRC     := hardware/camera_auth.c
HW_BIN     := hardware/camera_auth

# Default target
help:
	@echo ""
	@echo "  DeepShield KYC – Command Reference"
	@echo "  ────────────────────────────────────"
	@echo "  make install      Install Python dependencies"
	@echo "  make compile-hw   Compile C camera auth binary"
	@echo "  make dev          Run dev server (hot reload)"
	@echo "  make build        Build Docker image"
	@echo "  make docker-up    Start all services (Docker Compose)"
	@echo "  make docker-down  Stop all services"
	@echo "  make test         Run test suite"
	@echo "  make lint         Run ruff linter"
	@echo "  make clean        Remove cache & temp files"
	@echo "  make logs         Tail application logs"
	@echo ""

# ── Setup ──────────────────────────────────────────────────────────────────
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✓ Python dependencies installed"

compile-hw:
	@echo "Compiling C hardware auth binary..."
	gcc -O2 -o $(HW_BIN) $(HW_SRC)
	chmod +x $(HW_BIN)
	@echo "✓ Binary compiled: $(HW_BIN)"

setup: install compile-hw
	mkdir -p uploads models logs
	cp -n .env.example .env 2>/dev/null || true
	@echo "✓ Full setup complete"

# ── Development ───────────────────────────────────────────────────────────
dev:
	DEBUG=true $(UVICORN) $(APP) \
		--host 0.0.0.0 --port 8000 \
		--reload --log-level debug

dev-redis:
	docker run -d --name ds_redis -p 6379:6379 redis:7-alpine
	@echo "✓ Redis started on :6379"

# ── Docker ────────────────────────────────────────────────────────────────
build:
	docker build -t deepshield-kyc:latest .
	@echo "✓ Docker image built: deepshield-kyc:latest"

docker-up:
	docker-compose up -d
	@echo "✓ Services started"
	@echo "  App:   http://localhost"
	@echo "  API:   http://localhost/docs"
	@echo "  Redis: localhost:6379"

docker-down:
	docker-compose down
	@echo "✓ Services stopped"

docker-restart:
	docker-compose restart app
	@echo "✓ App restarted"

# ── Testing ───────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-fast:
	$(PYTHON) -m pytest tests/ -v --tb=short -x -q

test-cov:
	$(PYTHON) -m pytest tests/ --cov=backend --cov-report=html
	@echo "✓ Coverage report: htmlcov/index.html"

# ── Code quality ──────────────────────────────────────────────────────────
lint:
	$(PYTHON) -m ruff check backend/ --fix
	$(PYTHON) -m ruff format backend/
	@echo "✓ Linting complete"

# ── Utilities ─────────────────────────────────────────────────────────────
logs:
	docker-compose logs -f app

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
	rm -rf uploads/*.webm uploads/*.wav uploads/*_doc* 2>/dev/null || true
	@echo "✓ Cache cleaned"

# ── Download pretrained model checkpoint (FaceForensics++ fine-tuned) ──────
download-model:
	@echo "Downloading EfficientNet-B4 FF++ checkpoint..."
	@mkdir -p models
	@echo "NOTE: Place your fine-tuned checkpoint at models/deepfake_effnetb4_ff++.pth"
	@echo "      Or use a HuggingFace model — see README for links."
