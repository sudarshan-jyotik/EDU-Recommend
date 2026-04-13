# EduRecommend — BTP Presentation Script
## 12-Slide Deck with Speaker Notes

> Print this file. Each slide section gives you the exact content to put on the slide
> AND the speaker notes to say out loud. Practice until you can do it in 5 minutes.

---

## SLIDE 1 — Title

**Slide content:**
```
EduRecommend
A Personalised Course, Book & Study Plan
Recommender System Using Machine Learning

[Your Name] | Roll No. [XXX]
B.Tech CSE — Semester 4
BTP Project | [Year]
Guide: [Prof. Name]
```

**Speaker notes (30 seconds):**
"Good morning. My BTP project is EduRecommend — a machine learning system that gives engineering students personalised book recommendations, course recommendations, and a day-by-day study schedule — all from one web application, deployed publicly and accessible from any browser."

---

## SLIDE 2 — Problem Statement

**Slide content:**
```
The Problem

A 4th-sem student preparing for GATE or a web dev internship must
sift through thousands of resources to find what's relevant.

❌  General platforms (Amazon, Coursera) don't know your branch or exam goal
❌  No single tool recommends books + courses + study plans together
❌  Existing tools require login / history to personalise

EduRecommend solves all three.
```

**Speaker notes (30 seconds):**
"The core problem is information overload combined with lack of context. Amazon recommends books to everyone the same way. Coursera doesn't know if you're a CSE student preparing for GATE or a ME student learning Python for the first time. EduRecommend addresses this gap with three ML-powered modules in one system."

---

## SLIDE 3 — System Overview

**Slide content:**
```
Three ML Modules — One Unified System

┌─────────────────────────────────────────────┐
│            Student Profile                   │
│   Branch · Semester · Interests · Goal       │
└──────┬───────────────┬───────────────┬───────┘
       ▼               ▼               ▼
  Book Recs        Course Recs      Study Plan
  KNN + Cosine     TF-IDF +         NLTK +
  Similarity       Cosine Sim       Rule Engine
  270k+ Ratings    3,500 Courses    10 Domains

All served via FastAPI REST API · Deployed on Render.com
```

**Speaker notes (45 seconds):**
"The system has three independent ML modules. Books use item-based collaborative filtering — KNN with cosine similarity — on the Book-Crossing dataset with 270,000 books and 1.1 million ratings. Courses use TF-IDF content-based filtering on a Coursera dataset of 3,500 courses. Study plans use NLTK to extract keywords from the student's goal text and map them to a structured schedule. All three are exposed through a FastAPI REST API with automatic Swagger documentation."

---

## SLIDE 4 — Architecture Diagram

**Slide content:**
```
System Architecture

  Browser                    FastAPI (Docker)              Models
  ─────────                  ────────────────              ──────
  index.html ──fetch()──► /api/v1/books/recommend ──► KNN model
  courses.html ─fetch()──► /api/v1/courses/recommend ──► TF-IDF
  plan.html ───fetch()──► /api/v1/plan/generate ──► NLTK engine
  profile.html            /api/v1/feedback/       ──► feedback_log

  Frontend: Pure HTML + JS (no framework)
  Backend:  FastAPI + Pydantic + uvicorn
  Deploy:   Docker → Render.com
  CI:       GitHub Actions (pytest on every push)
```

**Speaker notes (45 seconds):**
"The architecture is deliberately decoupled. The frontend is pure HTML and JavaScript — no React, no framework. It calls the API with fetch() and renders JSON responses into the DOM. The backend is a FastAPI application that loads 7 pickle files at startup — the trained models — and serves them through typed endpoints. Everything runs in Docker. GitHub Actions runs the test suite on every push."

---

## SLIDE 5 — Book Recommender (ML Detail)

**Slide content:**
```
Book Recommender — KNN Collaborative Filtering

Dataset: Book-Crossing (271k books, 1.1M ratings)

Pipeline:
DataLoader → BookCleaner → UserCleaner → DataMerger
→ RatingFilter(≥50 ratings) → MatrixBuilder → KNNRecommender

Key decisions:
• Remove implicit ratings (71.4% of data) — value=0 means no preference
• Filter: users with ≥50 ratings, books with ≥50 ratings
• Cosine similarity (rating-magnitude independent)
• n_neighbors=11 (supports top-10 results)
• Bayesian average for popularity ranking
```

**Speaker notes (60 seconds):**
"The book recommender uses item-based collaborative filtering. I first explored the data and found that 71.4% of ratings are implicit — the user accessed the book but didn't rate it. Keeping these would train the model on access patterns, not preferences, so I remove them. I then filter to users with at least 50 ratings and books with at least 50 ratings — this reduces sparsity from 99.98% to a more manageable sub-matrix. I chose cosine similarity over Euclidean distance because it's magnitude-independent — a user who gives 8-9-10 to the same books as someone who gives 1-2-3 should still be considered similar. The entire pipeline is implemented as sklearn BaseEstimator transformers."

---

## SLIDE 6 — Course Recommender (ML Detail)

**Slide content:**
```
Course Recommender — TF-IDF Content-Based Filtering

Dataset: Coursera (3,500+ courses)

Process:
1. combined_text = title + subject + level (lowercase)
2. TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)
3. cosine_similarity(query_vector, all_course_vectors)
4. Optional: Level filter (Beginner/Intermediate/Advanced)
5. Optional: CSE branch boost (×1.2 for CS/Data courses)

Why TF-IDF over CF?
→ No user-course rating matrix available
→ Content-only features sufficient for topic matching
```

**Speaker notes (45 seconds):**
"Courses use content-based filtering because there is no user-course rating matrix — we don't know which users took which courses. TF-IDF converts each course's combined text into a vector where rare, informative words get higher weights. Bigrams capture phrases like 'machine learning' as a single feature — much better than treating 'machine' and 'learning' separately. The CSE branch boost is a simple hybrid: if the student's profile says CSE, Computer Science and Data Science courses get a 1.2x score multiplier."

---

## SLIDE 7 — Study Plan Generator

**Slide content:**
```
Study Plan Generator — NLTK Rule Engine

Input:  "prepare for GATE exam in 30 days"
Output: 30-day schedule with revision days + mock test

Pipeline:
1. word_tokenize("prepare for GATE exam in 30 days")
   → ['prepare', 'gate', 'exam', 'days']
2. Remove stopwords → top-5 keywords: ['gate', 'exam', 'prepare']
3. Keyword → domain: 'gate' → GATE syllabus (7 topics)
4. Distribute topics across 30 days
5. Insert revision day every 7th day
6. Final day = Full Mock Test

10 supported domains | 120+ topics | PDF export
```

**Speaker notes (30 seconds):**
"The study plan generator uses NLTK to extract keywords from a free-text goal, maps them to one of 10 pre-defined study domains — GATE, DSA, Web Dev, ML, Cloud, System Design, and more — and generates a day-by-day schedule. Revision days are automatically inserted every 7 days. The output can be exported as a PDF directly from the browser using jsPDF."

---

## SLIDE 8 — Demo Screenshots

**Slide content:**
```
[INSERT 4 SCREENSHOTS HERE]

1. Homepage — popular books grid with cover images and Bayesian scores
2. Book search results — similarity % bars, "Similar to this →" button
3. Course search — level filter chips, CSE boost badge, Enrol button
4. Study plan — vertical timeline with day dots, subtopics, PDF export button

Live demo: https://your-app.onrender.com
API docs:  https://your-app.onrender.com/docs
```

**Speaker notes (60 seconds):**
"[SHOW LIVE DEMO if internet available — otherwise use screenshots]
The homepage loads the top 50 books by Bayesian popularity. When I search for 'Harry Potter', the API returns 10 similar books with similarity percentage bars showing how close each recommendation is. Each card has a thumbs up/down which records feedback to the API. For courses, I can filter by Beginner/Intermediate/Advanced and if my profile says CSE, Computer Science courses rank higher. The study plan page takes my goal, generates a 30-day timeline, and lets me download it as a PDF. The Swagger UI at /docs shows every endpoint is auto-documented with try-it-out functionality."

---

## SLIDE 9 — Evaluation Results

**Slide content:**
```
Evaluation Results

Book Recommender (KNN — 80/20 split, 100 test users):
  Precision@5   = 0.22   (1.1 of top-5 are relevant)
  Hit Rate@10   = 0.41   (41% users found at least 1 relevant book)
  Relevance threshold: Rating ≥ 8/10

Course Recommender (TF-IDF — 10 manual test queries):
  Mean Precision@5 = 0.74
  Best: ML, Deep Learning = 1.00
  Lowest: Cybersecurity = 0.60

Study Plan (Unit tests — 30 test cases):
  Domain detection: 100% on 10 predefined goals
  Schedule correctness: All assertions pass
```

**Speaker notes (45 seconds):**
"For books, a precision@5 of 0.22 means on average just over 1 of the top 5 recommendations is a book the user actually rated highly. This is consistent with published benchmarks for collaborative filtering on sparse datasets — Herlocker et al. report 0.15–0.35 for MovieLens which is much denser than Book-Crossing. The hit rate of 0.41 at K=10 means 41% of users had at least one relevant book. For courses, the 0.74 precision is strong — the model consistently surfaces relevant courses for well-represented topics like Machine Learning and Python."

---

## SLIDE 10 — Production Engineering Highlights

**Slide content:**
```
What makes this production-grade

✓ FastAPI + Pydantic — auto-validates every request, auto-generates /docs
✓ Custom 422 handler — clean field-level error messages (not raw Pydantic)
✓ Dependency injection — models loaded once, injected into routes via Depends()
✓ Docker + docker-compose — anyone can run it in 2 commands
✓ GitHub Actions CI — pytest runs on every push (green badge on README)
✓ Feedback endpoint — logs relevance data for future model retraining
✓ Decoupled frontend — API testable independently of UI
✓ 100+ tests — test_api.py, test_plan.py, test_integration.py
```

**Speaker notes (30 seconds):**
"Beyond the ML algorithms, I want to highlight the engineering decisions that make this project suitable for a production environment. FastAPI auto-generates a Swagger UI from Python type annotations — any developer can test all endpoints without reading source code. Docker means the evaluator can run the entire system in two commands. GitHub Actions gives a CI badge — every code change is tested automatically. These are the practices that distinguish a portfolio project from a tutorial."

---

## SLIDE 11 — Limitations and Future Work

**Slide content:**
```
Limitations                    Future Work
──────────────────────         ─────────────────────────────
Cold-start problem             Neural CF (Two-Tower model)
  → Use popularity as fallback   → SVD / matrix factorisation

Sparsity (99.98%)              User authentication + DB
  → KNN limited on sparse data   → Store ratings, retrain monthly

Small course dataset           Mobile application
  → 3,500 courses only           → React Native wrapper

Manual domain mapping          A/B testing
  → 50 rules is not scalable     → Compare KNN vs Neural CF
```

**Speaker notes (30 seconds):**
"The main limitation is the cold-start problem for new users and the high sparsity of Book-Crossing. For future work, the most impactful improvement would be replacing KNN with matrix factorisation — SVD or ALS — which learns latent factor representations and handles sparsity better. Adding user authentication and a feedback database would enable real-time retraining as user preferences are recorded."

---

## SLIDE 12 — Conclusion

**Slide content:**
```
Conclusion

✓ Built 3 ML modules: KNN (books), TF-IDF (courses), NLTK (plans)
✓ Deployed at:  https://your-app.onrender.com
✓ API docs at:  https://your-app.onrender.com/docs
✓ GitHub:       github.com/YOUR_USERNAME/edu-recommend

Tech stack:
Python · FastAPI · scikit-learn · Pydantic · NLTK
pandas · Docker · GitHub Actions · Render.com

Thank you. Questions?
```

**Speaker notes (20 seconds):**
"To summarise — I built a three-module ML recommendation system, deployed it publicly on Render, containerised it with Docker, and set up CI with GitHub Actions. The API is fully documented and testable at /docs. I'm happy to take questions."

---

## Timing Guide

| Slide | Topic | Time |
|-------|-------|------|
| 1 | Title | 0:30 |
| 2 | Problem | 0:30 |
| 3 | Overview | 0:45 |
| 4 | Architecture | 0:45 |
| 5 | Book recommender | 1:00 |
| 6 | Course recommender | 0:45 |
| 7 | Study plan | 0:30 |
| 8 | Demo | 1:00 |
| 9 | Results | 0:45 |
| 10 | Engineering | 0:30 |
| 11 | Future work | 0:30 |
| 12 | Conclusion | 0:20 |
| **Total** | | **~8 min** |

> Aim for 5–6 minutes without demo, 8 minutes with live demo.
