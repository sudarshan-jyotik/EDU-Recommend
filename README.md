# EduRecommend

[![CI](https://github.com/sudarshan-jyotik/EDU-Recommend/actions/workflows/ci.yml/badge.svg)](https://github.com/sudarshan-jyotik/EDU-Recommend/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED)](https://docker.com)

> **Personalised book, course & study plan recommender** — B.Tech CSE Project  
> 3 ML/NLP modules · FastAPI REST API · Responsive Dark-Theme UI · Docker · Vercel & Render ready

---

## Overview

EduRecommend helps students discover relevant books, technical courses, and structured study schedules in one unified web application:

| Module | Algorithm & Improvements (v2) | Dataset | Scale |
|--------|-------------------------------|---------|-------|
| **Book Recommender** | Item-based KNN (cosine similarity) on per-user z-score normalised ratings, partial-word title matching, and rating-weighted re-ranking (70% cosine + 30% quality) | Book-Crossing | 270k+ raw books · 1.1M+ ratings (731 curated titles) |
| **Course Recommender** | TF-IDF (1–3 n-grams, weighted fields) + cosine similarity, CS synonym expansion (`ml`, `dsa`, `k8s`, etc.), CSE branch boost (1.2x), and rating-weighted re-ranking (65% cosine + 35% rating) | Coursera (Kaggle) | 3,416 unique courses |
| **Study Plan Generator** | NLTK keyword extraction (with regex fallback) + rule-based day-by-day scheduler with weekly revision days, final mock test, and PDF export (`jsPDF`) | Built-in curriculum | 10 domains · 120+ topics |

---

## Live Deployments

- **Vercel:** *(updated once deployed)*
- **Render:** *(updated once deployed)*

Both deployments serve the interactive web UI (`/`), Swagger API docs (`/docs`), ReDoc (`/redoc`), and health check (`/health`).

---

## Quickstart

### Option 1 — Local Python

```bash
git clone https://github.com/sudarshan-jyotik/EDU-Recommend.git
cd EDU-Recommend

python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements-dev.txt
python scripts/download_nltk.py

uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** for the web app and **http://localhost:8000/docs** for interactive Swagger API documentation.

> Pre-built portable model bundles (`models/books_bundle.npz`, `models/books_meta.json.gz`, `models/courses_bundle.json.gz`) are included in `models/`, so all three modules work immediately after cloning.

### Option 2 — Docker

```bash
docker-compose up --build
```

---

## Rebuilding ML Model Artifacts (Optional)

If you want to regenerate the model bundles from raw datasets:

1. Place Book-Crossing CSVs (`data/Books.csv`, `data/Ratings.csv`, `data/Users.csv`) in `data/` (`data/coursera_courses.csv` is already included).
2. Run the unified build script:
   ```bash
   python scripts/build_models.py --data-dir data --model-dir models
   ```
   Pass `--pickles` if you also want classic `.pkl` artifacts locally.

---

## Running Tests

```bash
# Full test suite (system, books, courses, study plan, feedback, integration)
pytest tests/ -v

# Fast mode without ML model files
SKIP_MODEL_TESTS=1 pytest tests/ -v
```

---

## API Endpoints

Full OpenAPI schemas and interactive testing are available at `/docs`.

### Books (`/api/v1/books`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/books/popular?n=50` | Top popular books by Bayesian weighted score |
| `POST` | `/api/v1/books/recommend` | Top-*n* similar books for a given title |
| `GET` | `/api/v1/books/autocomplete?q=` | Prefix + substring title suggestions |

```bash
curl -X POST http://localhost:8000/api/v1/books/recommend \
  -H "Content-Type: application/json" \
  -d '{"book_name": "Harry Potter and the Sorcerer'\''s Stone", "n": 5}'
```

### Courses (`/api/v1/courses`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/courses/popular?n=30` | Top rated Coursera courses |
| `POST` | `/api/v1/courses/recommend` | Courses matching topic, optional `level` (`Beginner`, `Intermediate`, `Advanced`, `Mixed`), and optional `branch` (`CSE` boost) |
| `GET` | `/api/v1/courses/autocomplete?q=` | Course title suggestions |
| `GET` | `/api/v1/courses/metadata` | Available levels and subjects |

```bash
curl -X POST http://localhost:8000/api/v1/courses/recommend \
  -H "Content-Type: application/json" \
  -d '{"topic": "ml", "level": "Intermediate", "branch": "CSE", "n": 5}'
```

### Study Plan (`/api/v1/plan`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/plan/generate` | Day-by-day study schedule from a free-text goal (`7–60` days, `1–10` hours/day) |
| `GET` | `/api/v1/plan/domains` | Supported study domains and topic counts |

### Feedback & System
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/feedback/` | Record user relevance feedback (1–5 rating) |
| `GET` | `/api/v1/feedback/stats` | Aggregated feedback metrics |
| `GET` | `/health` | Service health check |
| `GET` | `/api/v1/info` | Endpoint summary |

---

## Project Structure

```
EDU-Recommend/
├── api/
│   └── index.py                    Vercel ASGI entrypoint
├── app/
│   ├── main.py                     FastAPI app, CORS, error handlers, static mount
│   ├── config.py                   pydantic-settings configuration
│   ├── dependencies.py             Lazy + lifespan model loader
│   ├── recommenders/
│   │   ├── book_recommender.py     KNN collaborative filtering + rating re-ranking
│   │   ├── course_recommender.py   TF-IDF + synonym expansion + CSE boost
│   │   └── study_plan.py           NLTK keyword extraction + schedule generator
│   ├── routers/                    books, courses, plan, feedback API routes
│   └── schemas/                    Pydantic v2 request/response models
├── frontend/
│   ├── index.html                  Book discovery & popular books
│   ├── recommend.html              Bookmarkable book results page
│   ├── courses.html                Course search with level filters
│   ├── plan.html                   Study plan generator + PDF export
│   ├── profile.html                Student profile (branch, semester, interests, goal)
│   └── static/                     Shared dark-theme CSS & JS utilities
├── models/                         Portable compressed ML artifacts (< 1 MB total)
├── data/                           Coursera courses dataset
├── notebooks/                      EDA, pipeline, and evaluation notebooks + charts
├── scripts/
│   ├── build_models.py             Reproducible v2 model artifact builder
│   ├── download_nltk.py            NLTK tokenizer & stopwords installer
│   └── setup.sh                    Unix quickstart script
├── tests/                          Unit, API, and integration tests (pytest)
├── Dockerfile                      Container image (Render / local Docker)
├── docker-compose.yml              Local container orchestration
├── render.yaml                     Render Blueprint configuration
├── vercel.json                     Vercel routing & serverless configuration
├── requirements.txt                Production runtime dependencies
└── requirements-dev.txt            Development, test & notebook dependencies
```

---

## Deployment

### Render (Docker Web Service)
1. Push the repository to GitHub.
2. On [render.com](https://render.com), choose **New → Blueprint** (or **New → Web Service** with Docker runtime) and connect `sudarshan-jyotik/EDU-Recommend`.
3. Render automatically reads `render.yaml` and builds `Dockerfile` with health check `/health`.

### Vercel (Serverless FastAPI + Static CDN)
1. Import `sudarshan-jyotik/EDU-Recommend` on [vercel.com](https://vercel.com) or run `vercel --prod`.
2. Vercel uses `vercel.json` to serve `frontend/` from the edge CDN and route `/api/*`, `/health`, `/docs`, `/redoc`, and `/openapi.json` to `api/index.py`.

---

## Author

**Sudarshan Jyotik** — B.Tech Computer Science & Engineering  
GitHub: [@sudarshan-jyotik](https://github.com/sudarshan-jyotik)

## License

MIT
