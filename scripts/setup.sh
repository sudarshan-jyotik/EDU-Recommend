#!/usr/bin/env bash
# EduRecommend — one-command local setup
# Usage: chmod +x scripts/setup.sh && ./scripts/setup.sh

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[!] ${NC} $1"; }
info() { echo -e "${BLUE}[->]${NC} $1"; }
fail() { echo -e "${RED}[X] ${NC} $1"; exit 1; }

echo ""
echo "EduRecommend — Local Setup"
echo "=================================="

# Check Python
PY=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
MAJOR=$(echo $PY | cut -d. -f1)
MINOR=$(echo $PY | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
  fail "Python $PY found. Need Python 3.10 or higher."
fi
ok "Python $PY"

# Create venv
if [ ! -d ".venv" ]; then
  info "Creating virtual environment..."
  python3 -m venv .venv
  ok "Virtual environment created at .venv/"
fi

source .venv/bin/activate
ok "Virtual environment activated"

# Upgrade pip first (critical - old pip causes version conflicts)
info "Upgrading pip..."
pip install --upgrade pip --quiet
ok "pip upgraded"

# Install requirements
info "Installing dependencies (this takes 1-2 minutes)..."
pip install -r requirements-dev.txt --quiet
ok "All dependencies installed"

# Download NLTK data
info "Downloading NLTK language data..."
python3 scripts/download_nltk.py
ok "NLTK data downloaded"

# Create .env
[ ! -f ".env" ] && cp .env.example .env 2>/dev/null || true
[ ! -f ".env" ] && echo "MODEL_DIR=models
MAX_RECOMMENDATIONS=20
FEEDBACK_LOG_PATH=feedback_log.jsonl
DEBUG=false" > .env
ok ".env ready"

# Check model bundles
echo ""
echo "Checking ML model artifacts..."
MISSING=0
for f in books_bundle.npz books_meta.json.gz courses_bundle.json.gz; do
  if [ -f "models/${f}" ]; then
    ok "models/${f}"
  else
    warn "models/${f} — missing (run python scripts/build_models.py)"
    MISSING=$((MISSING+1))
  fi
done

echo ""
echo "=================================="
if [ $MISSING -gt 0 ]; then
  echo -e "${YELLOW}Setup done. $MISSING pickles missing.${NC}"
  echo ""
  echo "Run these notebooks to generate pickles:"
  echo "  jupyter notebook notebooks/02_books_pipeline.ipynb"
  echo "  jupyter notebook notebooks/04_courses_pipeline.ipynb"
  echo ""
  echo "Then:"
  echo "  source .venv/bin/activate"
  echo "  uvicorn app.main:app --reload --port 8000"
else
  echo -e "${GREEN}All ready!${NC}"
  echo ""
  echo "  source .venv/bin/activate"
  echo "  uvicorn app.main:app --reload --port 8000"
  echo ""
  echo "  http://localhost:8000        -> App"
  echo "  http://localhost:8000/docs   -> API docs"
fi
echo ""
