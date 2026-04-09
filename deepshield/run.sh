#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  DeepShield KYC  –  run.sh
#  Starts the server after checking prerequisites.
#  Usage: ./run.sh [--port 8000] [--prod]
# ═══════════════════════════════════════════════════════════

set -e

PORT=8000
PROD=false

# Parse args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift ;;
        --prod) PROD=true ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
RESET="\033[0m"

echo -e "${CYAN}  DeepShield KYC  ·  Starting server...${RESET}"
echo ""

# ── Activate venv if it exists ─────────────────────────────────────────────
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}[OK]${RESET}   Virtual environment activated"
fi

# ── Check .env ────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}[WARN]${RESET} .env not found — copying from .env.example"
    cp .env.example .env
fi

# ── Check required dirs ───────────────────────────────────────────────────
mkdir -p uploads models logs

# ── Try to start Redis if not running ─────────────────────────────────────
if ! nc -z localhost 6379 2>/dev/null; then
    echo -e "${YELLOW}[WARN]${RESET} Redis not running on :6379"
    if command -v docker &>/dev/null; then
        echo -e "${CYAN}[INFO]${RESET} Starting Redis via Docker..."
        docker run -d --rm --name ds_redis -p 6379:6379 redis:7-alpine > /dev/null 2>&1 && \
            echo -e "${GREEN}[OK]${RESET}   Redis started" || \
            echo -e "${YELLOW}[WARN]${RESET} Could not start Redis — using in-memory fallback"
    else
        echo -e "${YELLOW}[WARN]${RESET} Docker not found — using in-memory session store"
    fi
else
    echo -e "${GREEN}[OK]${RESET}   Redis running on :6379"
fi

echo ""
echo -e "${GREEN}  Starting FastAPI server on http://localhost:${PORT}${RESET}"
echo -e "  API docs:  http://localhost:${PORT}/docs"
echo -e "  Health:    http://localhost:${PORT}/health"
echo ""
echo -e "  Press Ctrl+C to stop"
echo ""

# ── Start server ──────────────────────────────────────────────────────────
if [ "$PROD" = true ]; then
    python3 -m uvicorn backend.main:app \
        --host 0.0.0.0 \
        --port "$PORT" \
        --workers 4 \
        --log-level info
else
    python3 -m uvicorn backend.main:app \
        --host 0.0.0.0 \
        --port "$PORT" \
        --reload \
        --log-level debug
fi
