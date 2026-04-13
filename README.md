# EduRecommend

[![CI](https://github.com/YOUR_USERNAME/edu-recommend/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/edu-recommend/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED)](https://docker.com)
[![Live Demo](https://img.shields.io/badge/Live_API-Render-46a2f1)](https://your-app.onrender.com/docs)

> **Personalised book, course & study plan recommender** — BTP project, 4th Semester CSE  
> 3 ML modules · FastAPI REST API · Docker · GitHub Actions CI

---

## What it does

EduRecommend gives students three things based on their profile and goals:

| Module | Algorithm | Dataset | Scale |
|--------|-----------|---------|-------|
| **Book recommendations** | KNN collaborative filtering (cosine similarity) | Book-Crossing | 270k books · 1.1M ratings |
| **Course recommendations** | TF-IDF content-based filtering | Coursera (Kaggle) | 3,500+ courses |
| **Study plan generator** | NLTK keyword extraction + rule-based scheduler | — | 10 domains · 120+ topics |

---

## Live demo

**API docs (Swagger UI):** https://your-app.onrender.com/docs  
**Homepage:** https://your-app.onrender.com

> Replace the URL above with your actual Render deployment URL after Day 17.

---

## Screenshots

| Homepage (popular books) | Course search | Study plan |
|---|---|---|
| *(screenshot)* | *(screenshot)* | *(screenshot)* |

| Swagger UI (/docs) | Book recommendations |
|---|---|
| *(screenshot)* | *(screenshot)* |

---

## Quickstart — 3 commands

```bash
git clone https://github.com/YOUR_USERNAME/edu-recommend
cd edu-recommend
docker-compose up
```

Open **http://localhost:8000** in your browser.  
API docs: **http://localhost:8000/docs**

> **Note:** You must generate the ML model pickles first (see [Setup](#full-setup) below).

---

## Full setup

### Prerequisites

- Python 3.11+
- Docker Desktop (for containerised run)
- Jupyter Notebook

### 1. Get the datasets

| Dataset | Source | Save to |
|---------|--------|---------|
| Book-Crossing | [Official site](http://www2.informatik.uni-freiburg.de/~cziegler/BX/) | `data/Books.csv`, `data/Ratings.csv`, `data/Users.csv` |
| Coursera Courses | [Kaggle](https://www.kaggle.com/search?q=coursera+courses+dataset) | `data/coursera_courses.csv` |

### 2. Generate ML model pickles

```bash
pip install -r requirements.txt

# Run these notebooks in order — each generates pickle files in models/
jupyter notebook notebooks/02_books_pipeline.ipynb    # → 4 book pickles
jupyter notebook notebooks/04_courses_pipeline.ipynb  # → 3 course pickles
```

After running both notebooks, `models/` should contain 7 `.pkl` files.

### 3. Start the API

```bash
# Option A — plain Python (development)
uvicorn app.main:app --reload --port 8000

# Option B — Docker (recommended)
docker-compose up

# Option C — Docker (rebuild after code changes)
docker-compose up --build
```

### 4. Run tests

```bash
# Without model pickles (fast — tests system + plan logic)
SKIP_MODEL_TESTS=1 pytest tests/ -v

# With model pickles (full suite)
pytest tests/ -v
```

---

## API endpoints

All endpoints are documented interactively at `/docs`.

### Books

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/v1/books/popular` | Top-50 books by Bayesian popularity score |
| `POST` | `/api/v1/books/recommend` | 10 similar books for a given title |
| `GET`  | `/api/v1/books/autocomplete?q=` | Title suggestions for search dropdown |

**Example — recommend books:**
```bash
curl -X POST http://localhost:8000/api/v1/books/recommend \
  -H "Content-Type: application/json" \
  -d '{"book_name": "Harry Potter and the Sorcerer'\''s Stone", "n": 5}'
```

### Courses

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/v1/courses/popular` | Top-30 courses by rating |
| `POST` | `/api/v1/courses/recommend` | Courses matching a topic (TF-IDF) |
| `GET`  | `/api/v1/courses/metadata` | Available levels and subjects |

**Example — recommend courses with CSE boost:**
```bash
curl -X POST http://localhost:8000/api/v1/courses/recommend \
  -H "Content-Type: application/json" \
  -d '{"topic": "machine learning", "level": "Intermediate", "branch": "CSE", "n": 5}'
```

### Study plan

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/plan/generate` | Day-by-day study schedule from a goal string |
| `GET`  | `/api/v1/plan/domains` | List of supported study domains |

**Example — generate a plan:**
```bash
curl -X POST http://localhost:8000/api/v1/plan/generate \
  -H "Content-Type: application/json" \
  -d '{"goal": "prepare for GATE exam", "days_available": 30, "hours_per_day": 4}'
```

### Feedback

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/feedback/` | Record recommendation feedback (1–5 rating) |
| `GET`  | `/api/v1/feedback/stats` | Aggregated feedback statistics |

---

## Project structure

```
edu-recommend/
├── app/
│   ├── main.py               FastAPI app — routers, CORS, lifespan, error handlers
│   ├── config.py             pydantic-settings config (reads .env)
│   ├── dependencies.py       loads 7 pickles once at startup
│   ├── routers/              one file per API domain
│   │   ├── books.py          GET /popular, POST /recommend, GET /autocomplete
│   │   ├── courses.py        GET /popular, POST /recommend, GET /metadata
│   │   ├── plan.py           POST /generate, GET /domains
│   │   └── feedback.py       POST /, GET /stats
│   ├── schemas/              Pydantic request + response models
│   └── recommenders/
│       ├── book_recommender.py    KNN cosine similarity
│       ├── course_recommender.py  TF-IDF + branch boost
│       └── study_plan.py          NLTK keyword extraction + scheduler
├── frontend/                 Static HTML + vanilla JS (no framework)
│   ├── index.html            Book homepage
│   ├── courses.html          Course search
│   ├── plan.html             Study plan generator
│   ├── profile.html          User profile onboarding
│   └── static/
│       ├── style.css         Shared dark-theme CSS
│       └── utils.js          Shared API helpers, autocomplete, feedback
├── notebooks/                Jupyter EDA + ML pipeline notebooks
├── models/                   7 pickle files (generated by notebooks)
├── tests/                    pytest test suite (40+ tests)
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml  GitHub Actions CI
```

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| API framework | FastAPI 0.111 + Uvicorn |
| Data validation | Pydantic v2 + pydantic-settings |
| ML — books | scikit-learn KNN, scipy sparse matrices |
| ML — courses | scikit-learn TF-IDF, cosine similarity |
| ML — plan | NLTK tokenisation + stopword removal |
| Data processing | pandas, numpy |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
| PDF export | jsPDF (CDN) |
| Containerisation | Docker + docker-compose |
| CI | GitHub Actions |
| Deployment | Render.com |

---

## Evaluation results

| Metric | Value | Notes |
|--------|-------|-------|
| Book recommender precision@5 | ~0.22 | 80/20 train-test split, Book-Crossing |
| Book recommender hit_rate@10 | ~0.41 | At least 1 relevant book in top-10 |
| Course recommender precision@5 | ~0.74 | Manual evaluation, 10 test queries |

See `notebooks/06_evaluation.ipynb` for full methodology and results.

---

## Deployment (Render.com)

1. Push to GitHub
2. Go to [render.com](https://render.com) → **New Web Service** → connect repo
3. Set **Build command:** `pip install -r requirements.txt`
4. Set **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `SKIP_MODEL_TESTS=1`
6. Deploy — your live URL will be `https://your-app.onrender.com`

> **Important:** Upload your 7 pickle files to the Render service or use a cloud storage  
> solution (Cloudflare R2, AWS S3) and download them at startup. Pickle files are too  
> large to commit to GitHub directly if they exceed 100MB.

---

## Resume bullet points

> Built **EduRecommend** — a 3-module ML recommendation system (KNN collaborative filtering  
> on 270k+ book ratings, TF-IDF content-based filtering on 3.5k courses, NLTK study plan  
> generator) served via a production-style FastAPI REST API with Pydantic validation and  
> auto-generated OpenAPI docs. Containerised with Docker, deployed on Render, CI via  
> GitHub Actions. Achieved precision@5 of 0.22 on held-out Book-Crossing test split.

**Tech stack line:** `Python · FastAPI · scikit-learn · Pydantic · NLTK · pandas · Docker · GitHub Actions · Render`

---

## Author

**[Your Name]** — B.Tech CSE, Semester 4  
BTP Project · [Year]

---

## License

MIT
