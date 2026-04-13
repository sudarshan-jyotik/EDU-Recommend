FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (none needed for pre-built wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data at build time (faster container startup)
RUN python -c "\
import nltk; \
nltk.download('punkt',     quiet=True); \
nltk.download('punkt_tab', quiet=True); \
nltk.download('stopwords', quiet=True); \
print('NLTK data ready')" || echo "NLTK download failed - fallback will be used"

COPY app/      ./app/
COPY frontend/ ./frontend/
COPY scripts/  ./scripts/
COPY models/   ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
