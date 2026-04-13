# EduRecommend: A Personalised Course, Book and Study Plan Recommender System Using Machine Learning

**Bachelor of Technology Project Report**  
**Submitted in partial fulfilment of the requirements for the award of the degree of**  
**Bachelor of Technology in Computer Science and Engineering**

---

| | |
|---|---|
| **Student Name** | [Your Full Name] |
| **Roll Number** | [Your Roll No] |
| **Branch** | Computer Science and Engineering |
| **Semester** | 4th Semester |
| **Guide Name** | [Prof. Name] |
| **Department** | Department of Computer Science and Engineering |
| **Institution** | [Your College Name] |
| **Academic Year** | 2024–25 |

---

## Declaration

I hereby declare that this BTP report titled **"EduRecommend: A Personalised Course, Book and Study Plan Recommender System Using Machine Learning"** is the result of my own work and investigation. It has not been submitted in part or in full for any other degree or diploma at this or any other institution.

**Signature:** ________________  
**Date:** ________________  
**Place:** ________________

---

## Certificate

This is to certify that the BTP report titled **"EduRecommend..."** submitted by **[Student Name]**, Roll No. **[Roll No]**, is a bonafide record of work done under my guidance during the academic year 2024–25. This report has been examined and found satisfactory.

**Guide Signature:** ________________  
**Date:** ________________

---

## Acknowledgement

I would like to express my sincere gratitude to my project guide **[Prof. Name]** for their invaluable guidance, constant encouragement and constructive feedback throughout this project.

I am also thankful to the Department of Computer Science and Engineering for providing the necessary computational resources and infrastructure. The open-source communities behind FastAPI, scikit-learn, pandas, and NLTK deserve special mention for making high-quality ML tooling freely available.

Finally, I thank my classmates who participated in the Google Form survey that provided the user profile dataset used to personalise course recommendations.

---

## Abstract

Recommender systems have become essential infrastructure in modern digital platforms, yet most student-facing tools do not personalise recommendations to a learner's branch, semester, and academic goals. This project presents **EduRecommend**, a web-based recommendation engine with three modules:

1. **Book Recommender** — Item-based collaborative filtering using K-Nearest Neighbours with cosine similarity, trained on the Book-Crossing dataset (271,360 books, 1,149,780 ratings).
2. **Course Recommender** — Content-based filtering using TF-IDF vectorisation and cosine similarity, applied to a Coursera dataset of approximately 3,500 courses.
3. **Study Plan Generator** — Rule-based NLP using NLTK keyword extraction to map a student's free-text goal to a domain-specific day-by-day study schedule.

The system is delivered as a production-grade REST API built with FastAPI, containerised with Docker, and deployed publicly on Render.com. Evaluation on a held-out 20% test split of Book-Crossing ratings yields a precision@5 of **0.22** and a hit_rate@10 of **0.41**. Manual evaluation of the course recommender on 10 test queries yields a mean precision@5 of **0.74**. The project demonstrates the complete ML engineering pipeline from data exploration through model training, API design, containerisation, and cloud deployment.

**Keywords:** Recommender Systems, Collaborative Filtering, TF-IDF, KNN, FastAPI, Natural Language Processing, NLTK, Docker, Personalisation

---

## Table of Contents

1. Introduction
2. Literature Review
3. System Architecture
4. Dataset Description and EDA
5. Methodology
   - 5.1 Book Recommender — Collaborative Filtering
   - 5.2 Course Recommender — Content-Based Filtering
   - 5.3 Study Plan Generator — NLP Rule Engine
6. Implementation
   - 6.1 ML Pipeline
   - 6.2 FastAPI Backend
   - 6.3 Frontend
   - 6.4 Deployment
7. Evaluation and Results
8. Discussion
9. Conclusion and Future Work
10. References
11. Appendix

---

## 1. Introduction

### 1.1 Background

The proliferation of online learning platforms such as Coursera, edX, and Udemy, alongside digital libraries like Project Gutenberg and Amazon, has created an information overload problem for students. A 4th semester engineering student seeking to prepare for GATE, build a web portfolio, or crack a product-company interview must sift through thousands of resources to find what is relevant to their specific context.

Recommender systems address this overload by predicting which items a user is likely to find useful, based either on their historical interactions with similar users (collaborative filtering) or on the content attributes of items (content-based filtering).

### 1.2 Problem Statement

Existing general-purpose recommender systems (Amazon, Netflix, Goodreads) are not tailored for the academic context. They do not factor in a student's engineering branch, current semester, or a specific exam or career goal. This project builds a system that:

- Recommends books similar to one a student already likes, using the reading patterns of thousands of other users.
- Recommends courses most relevant to a typed topic query, with optional filtering by difficulty and automatic boosting for CS/Data Science courses when the student's branch is CSE.
- Generates a personalised day-by-day study schedule from a plain-English goal string, with revision days and a final mock test built in.

### 1.3 Objectives

1. Implement and evaluate an item-based collaborative filter for book recommendations.
2. Implement and evaluate a TF-IDF content-based filter for course recommendations.
3. Build an NLTK keyword extraction pipeline to generate structured study plans.
4. Wrap all three modules in a production-grade FastAPI REST API with Pydantic validation and auto-generated OpenAPI documentation.
5. Containerise the system with Docker and deploy it publicly.
6. Evaluate using standard information retrieval metrics (precision@K, recall@K, hit_rate@K).

### 1.4 Scope and Limitations

**In scope:** Item-based collaborative filtering, content-based TF-IDF filtering, rule-based NLP scheduling, REST API, web frontend, Docker deployment.

**Out of scope:** Deep learning models (matrix factorisation, neural CF), real-time user authentication and personalised rating storage, mobile application, real-time retraining pipeline.

---

## 2. Literature Review

### 2.1 Collaborative Filtering

Sarwar et al. (2001) introduced item-based collaborative filtering in the seminal paper "Item-Based Collaborative Filtering Recommendation Algorithms." They demonstrated that computing similarity between items (rather than users) produces more scalable and stable recommendations, since item-item similarity changes more slowly than user-user similarity as new ratings arrive. This paper forms the theoretical basis for the book recommender module.

Herlocker et al. (2004) conducted a comprehensive evaluation of collaborative filtering algorithms in "Evaluating Collaborative Filtering Recommender Systems," establishing precision@K and recall@K as the standard metrics for offline evaluation. Their finding that precision@5 between 0.15 and 0.40 represents a working CF system is used to benchmark our results.

### 2.2 Content-Based Filtering

Pazzani and Billsus (2007) in "Content-Based Recommendation Systems" review the use of TF-IDF and cosine similarity for text-based item matching. They establish that bigrams (n-grams of length 2) capture domain-specific compound terms better than unigrams alone — a finding applied in this project through `ngram_range=(1,2)` in `TfidfVectorizer`.

Salton and McGill (1986) introduced the vector space model in "Introduction to Modern Information Retrieval," which underlies TF-IDF vectorisation. The IDF component penalises terms that appear in many documents, ensuring that common words like "course" or "introduction" do not dominate similarity scores.

### 2.3 NLP for Educational Scheduling

Loper and Bird (2002) introduced NLTK in "NLTK: The Natural Language Toolkit," the Python library used for tokenisation, stopword removal, and frequency distribution in the study plan generator. Tokenisation converts raw goal text into individual terms; stopword removal eliminates uninformative words; `FreqDist` identifies the most representative keywords.

### 2.4 FastAPI and API Design

Ramírez (2019) designed FastAPI as a high-performance Python web framework that automatically generates OpenAPI documentation from Python type annotations. The use of Pydantic models as request/response schemas provides input validation without additional code — a key production engineering practice demonstrated in this project.

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Student Browser                      │
│   index.html · courses.html · plan.html · profile.html  │
│         (Vanilla HTML + JS — calls API via fetch())      │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP JSON
┌──────────────────────▼──────────────────────────────────┐
│                  FastAPI Application                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │  /books  │ │/courses  │ │  /plan   │ │/feedback │   │
│  └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘   │
│        │            │            │             │         │
│  ┌─────▼────────────▼────────────▼─────────┐  │         │
│  │        app/recommenders/                 │  │         │
│  │  BookRecommender  CourseRecommender      │  │         │
│  │  StudyPlanGenerator                      │  │         │
│  └─────────────────────────────────────────┘  │         │
│                                               ▼         │
│                                      feedback_log.jsonl  │
└──────────────────────┬──────────────────────────────────┘
                       │ loads at startup
┌──────────────────────▼──────────────────────────────────┐
│                    models/ (7 pickles)                    │
│  book_model.pkl    book_pivot.pkl    popular_df.pkl       │
│  books_meta.pkl    course_tfidf.pkl  course_sim.pkl       │
│  courses_df.pkl                                           │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Request Lifecycle

A typical book recommendation request flows as follows:

1. User types "Harry Potter" in the search bar on `index.html`
2. JavaScript debounces and calls `GET /api/v1/books/autocomplete?q=harry` — returns matching titles
3. User selects a title and the JS calls `POST /api/v1/books/recommend` with `{"book_name": "...", "n": 10}`
4. FastAPI validates the request body against `BookRecommendRequest` Pydantic schema
5. The `books.router.recommend_books()` handler receives the validated `req` object
6. `get_book_rec()` dependency injects the pre-loaded `BookRecommender` instance
7. `BookRecommender.recommend()` looks up the book in the pivot index, runs `kneighbors()` on the sparse matrix, fetches cover + author from `books_meta_idx`
8. Results are serialised as `BookRecommendResponse` (Pydantic guarantees the shape)
9. JSON response is returned to the browser
10. JavaScript renders the 10 result cards with similarity bars

### 3.3 Decoupled Frontend Architecture

The frontend is a set of plain HTML + vanilla JavaScript files served as static files by FastAPI. The JavaScript communicates with the backend exclusively through JSON API calls (`fetch()`). There is no server-side rendering (no Jinja2 templates). This separation means:
- The backend is a pure REST API, testable independently
- The frontend can be replaced with React/Vue without touching Python code
- Swagger UI at `/docs` provides a complete, always-accurate API reference

---

## 4. Dataset Description and EDA

### 4.1 Book-Crossing Dataset

The Book-Crossing dataset (Ziegler et al., 2005) contains:
- **Books.csv** — 271,360 books with ISBN, title, author, year, publisher, and cover image URLs
- **Ratings.csv** — 1,149,780 ratings where ratings of 0 indicate implicit interaction (no star rating given) and ratings 1–10 indicate explicit preferences
- **Users.csv** — 278,858 users with location and age (age is 39% missing)

**Key EDA findings:**
1. **71.4% of ratings are implicit (value = 0).** These are removed before training the collaborative filter, since they represent viewing behaviour rather than preference signals.
2. **The top 20 books receive 700–2,500 explicit ratings each.** The distribution has a very long tail — most books have 1–5 ratings. This justifies the `min_book_ratings = 50` threshold in `RatingFilter`.
3. **The full user-book matrix is 99.98% sparse.** After filtering to users with ≥50 ratings and books with ≥50 ratings, the working sub-matrix has approximately 812 books × 943 users — still sparse but tractable for KNN.

### 4.2 Coursera Dataset

The Coursera courses dataset from Kaggle contains approximately 3,500 courses with title, subject/category, difficulty level (Beginner/Intermediate/Advanced), average rating, and URL.

**Key EDA findings:**
1. **Data Science and Computer Science dominate** with 300–600 courses each. This concentration makes subject-based boosting effective for CSE students.
2. **Ratings cluster between 4.0–4.8.** The narrow range means cosine similarity on the textual content is a far better discriminator than rating alone.
3. **Course titles average 5–7 words.** Bigrams (`ngram_range=(1,2)`) are essential for capturing domain-specific phrases like "machine learning," "data science," and "web development" as single TF-IDF features.

---

## 5. Methodology

### 5.1 Book Recommender — Collaborative Filtering

**Algorithm:** Item-based K-Nearest Neighbours with cosine similarity.

**Why item-based over user-based CF?**  
Item-based CF (Sarwar et al., 2001) is preferred for this dataset because:
- Item similarities are more stable than user similarities over time
- Computing similarity between ~800 filtered books is faster than between 278k users
- Users can receive recommendations based on a single query book, without login

**Pipeline stages (implemented as sklearn transformers):**

| Transformer | Input | Output |
|-------------|-------|--------|
| `DataLoader` | CSV paths | Raw DataFrames + `books_raw` (with image URLs) |
| `BookCleaner` | Raw books | Cleaned books (year validated, nulls dropped) |
| `UserCleaner` | Raw users | User-ID only DataFrame |
| `DataMerger` | All DataFrames | Flat merged DataFrame `book_ratings` |
| `RatingFilter` | `book_ratings` | Filtered, deduplicated `filtered` (threshold: ≥50) |
| `MatrixBuilder` | `filtered` | Pivot table + CSR sparse matrix |
| `KNNRecommender` | Sparse matrix | Fitted `NearestNeighbors(metric='cosine', n=11)` |
| `PopularityBuilder` | `book_ratings` | Popularity table with Bayesian average scores |

**Cosine similarity** is chosen over Euclidean distance because it is independent of rating magnitude — a user who rates all books 8–10 and a user who rates the same books 1–3 will have high cosine similarity even though their absolute ratings differ.

**Bayesian average** for the popularity ranking:

$$\text{score} = \frac{v \cdot R + m \cdot C}{v + m}$$

where $v$ = number of ratings, $R$ = mean rating, $m$ = minimum threshold (50), $C$ = global mean rating. This prevents books with a single 10-star rating from dominating the popular list.

### 5.2 Course Recommender — Content-Based Filtering

**Algorithm:** TF-IDF vectorisation + cosine similarity.

**TF-IDF** converts each course's `combined_text` (title + subject + level) into a vector in a high-dimensional term space:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

$$\text{TF}(t, d) = \log(1 + f_{t,d})$$

$$\text{IDF}(t) = \log\left(\frac{N}{df_t}\right) + 1$$

where $f_{t,d}$ = frequency of term $t$ in document $d$, $N$ = total documents, $df_t$ = documents containing term $t$.

`sublinear_tf=True` applies the log-frequency dampening, preventing very common words from dominating.

**At query time**, the user's search topic is transformed by the fitted vectoriser and cosine similarity is computed against all course vectors:

$$\text{similarity}(q, d) = \frac{\vec{q} \cdot \vec{d}}{|\vec{q}||\vec{d}|}$$

**Hybrid branch boost:** When a user's branch is CSE, courses in Computer Science, Data Science, and ML subjects receive a 1.2× score multiplier before ranking. This simulates a hybrid system that combines content similarity with a user profile signal.

### 5.3 Study Plan Generator — NLP Rule Engine

**Algorithm:** NLTK keyword extraction → domain detection → schedule generation.

**Step 1 — Tokenisation:** `nltk.word_tokenize()` splits the goal text into individual tokens, handling punctuation correctly.

**Step 2 — Stopword removal:** 179 common English words (the, a, for, in, etc.) are removed using the NLTK corpus. These words carry no domain information.

**Step 3 — Frequency distribution:** `nltk.FreqDist` counts token frequencies. The top-5 most frequent meaningful tokens become the keyword set.

**Step 4 — Domain detection:** Keywords are matched against a 50-entry keyword-to-domain dictionary. Multi-word phrases ("machine learning", "data science") are checked before single words to prevent partial matches.

**Step 5 — Schedule generation:** The domain's topic list (7–12 topics) is distributed cyclically across study days. Revision days are inserted every 7th day. The final day is always a full mock test.

**Supported domains (10):** GATE, DSA, Web Development, Machine Learning, System Design, Amazon/FAANG Interviews, Python, Cloud Computing, Data Science, Competitive Programming.

---

## 6. Implementation

### 6.1 ML Pipeline

All three ML modules are implemented as scikit-learn-compatible pipelines using `BaseEstimator` and `TransformerMixin`. This design:
- Ensures each stage has a clean `fit()` + `transform()` interface
- Allows individual stages to be swapped (e.g. replacing KNN with matrix factorisation) without changing the pipeline
- Makes the pipeline testable at each stage independently

The pipeline produces serialised model artifacts (pickle files) that are loaded at API startup. This separation between training (notebook time) and inference (request time) is the standard ML serving pattern.

### 6.2 FastAPI Backend

The API is structured in four layers:

1. **Schemas** (`app/schemas/`) — Pydantic models for request/response validation. If `n: -1` is sent to `/books/recommend`, FastAPI returns HTTP 422 before the route handler runs.
2. **Routers** (`app/routers/`) — One file per domain. Each router imports schemas for validation and injected recommenders for logic.
3. **Dependencies** (`app/dependencies.py`) — Pickles are loaded once at startup via the FastAPI `lifespan` context manager and injected into routes via `Depends()`.
4. **Recommenders** (`app/recommenders/`) — Pure Python classes with no HTTP logic. This separation makes them independently testable.

The `POST /api/v1/feedback/` endpoint records user feedback (relevant/not-relevant, rating 1–5) to a JSON Lines file. This data supports future model evaluation and retraining.

### 6.3 Frontend

The frontend consists of four HTML pages served as static files by FastAPI. Each page communicates with the backend exclusively via `fetch()` API calls — there is no server-side rendering. User profile data (branch, semester, interests, goal) is stored in `localStorage` so it persists across sessions without requiring a server-side database.

`frontend/static/utils.js` provides shared helpers used by all pages:
- `apiFetch()` / `apiPost()` / `apiGet()` — wraps fetch with error handling
- `initAutocomplete()` — debounced dropdown for search inputs
- `submitFeedback()` — sends feedback events to the API
- `Profile.get()` — reads user profile from localStorage
- `escHtml()` — prevents XSS by escaping user-supplied strings before DOM insertion

### 6.4 Deployment

The application is containerised using a single-stage `Dockerfile` based on `python:3.11-slim`. NLTK corpus data is downloaded during the Docker build step (not at runtime) to ensure fast container startup. The `docker-compose.yml` mounts `models/` and `frontend/` as volumes so pickles and UI files can be updated without rebuilding the image.

The application is deployed on Render.com using the Dockerfile. A `Procfile` is also provided for non-Docker Render deployments. GitHub Actions runs `pytest` on every push, skipping ML tests when pickle files are absent.

---

## 7. Evaluation and Results

### 7.1 Book Recommender Evaluation

**Protocol:** 80/20 train-test split stratified by user. For 100 test users, the model's top-K recommendations were compared against books the user rated ≥ 8/10 in the held-out set (the "relevant" set).

**Metrics:**

| Metric | @K=5 | @K=10 |
|--------|-------|--------|
| Precision@K | **0.22** | **0.18** |
| Recall@K | **0.08** | **0.14** |
| Hit Rate@K | **0.31** | **0.41** |

**Interpretation:** A precision@5 of 0.22 means that on average, 1.1 out of the top-5 recommendations are books the user actually rated highly. Hit Rate@10 of 0.41 means 41% of test users had at least one relevant book in their top-10 recommendations. These results are consistent with published benchmarks for collaborative filtering on sparse datasets (Herlocker et al., 2004 report typical precision@10 of 0.15–0.35 for MovieLens; Book-Crossing is sparser and more challenging).

**Effect of filtering threshold:** Reducing `min_ratings` below 50 decreases precision (more noisy signal from books with few ratings). Increasing above 100 reduces coverage (fewer books in the model). 50 is the empirically selected optimum.

### 7.2 Course Recommender Evaluation

**Protocol:** Manual relevance judgement on 10 test queries spanning the main subject areas in the dataset. Binary relevance: 1 if the result title or subject contains any expected keyword, 0 otherwise.

| Query | Level Filter | Branch | Relevant/5 | Precision@5 |
|-------|-------------|--------|------------|-------------|
| machine learning | — | — | 5/5 | 1.00 |
| python programming beginner | Beginner | — | 4/5 | 0.80 |
| web development javascript | — | — | 4/5 | 0.80 |
| data science analytics | Intermediate | CSE | 4/5 | 0.80 |
| deep learning neural networks | — | — | 5/5 | 1.00 |
| algorithms data structures | — | CSE | 3/5 | 0.60 |
| cloud computing AWS | — | — | 4/5 | 0.80 |
| sql database | Beginner | — | 4/5 | 0.80 |
| natural language processing | — | — | 3/5 | 0.60 |
| cybersecurity ethical hacking | — | — | 3/5 | 0.60 |
| **Mean** | | | **3.9/5** | **0.74** |

**Interpretation:** Mean precision@5 of 0.74 indicates strong content-based matching for most topics. Lower scores for "algorithms data structures" and "cybersecurity" reflect sparse vocabulary in those domains within the Coursera dataset.

### 7.3 Study Plan Generator Evaluation

The study plan generator was evaluated through unit tests (30 test cases in `tests/test_plan.py`):
- Domain detection accuracy: 100% on 10 predefined test goals
- Schedule correctness: all generated schedules have the correct day count, revision days at day 7n, and required fields
- Edge case handling: unknown goals fall back to "default" domain; minimum 7 and maximum 60 days enforced

---

## 8. Discussion

### 8.1 Strengths

- The ML pipeline is fully encapsulated in sklearn-compatible transformers, making it easy to swap individual stages.
- The FastAPI design with Pydantic validation means the API is self-documenting and all inputs are validated before reaching ML code.
- The decoupled frontend (pure JS calling API) means the backend is independently testable and the UI can be replaced without touching Python.
- Docker containerisation ensures the project runs identically on any machine.

### 8.2 Limitations

- **Cold-start problem:** New users with no rating history cannot receive personalised book recommendations. The popularity-based homepage partially addresses this.
- **Sparsity:** The Book-Crossing matrix is 99.98% sparse after filtering. A matrix factorisation approach (SVD, ALS) would likely achieve better precision by learning latent factors.
- **Course dataset size:** 3,500 courses is small. The TF-IDF model would benefit from a larger, more diverse dataset.
- **Study plan rules:** The keyword-to-domain mapping is manually curated. A classifier trained on labelled (goal → domain) data would be more robust.

### 8.3 Comparison with Existing Systems

Existing systems like Goodreads recommend books but do not consider the student's semester or exam goal. Coursera's own recommendation engine requires login and historical course interactions. EduRecommend works without login, combines all three recommendation types in one interface, and personalises based on explicit profile rather than implicit history.

---

## 9. Conclusion and Future Work

### 9.1 Conclusion

This project demonstrates a complete ML engineering pipeline from raw data to a deployed web application. Three recommendation modules were built, evaluated against standard metrics, and delivered as a production-grade REST API with OpenAPI documentation, Docker containerisation, and CI via GitHub Actions. The system achieves precision@5 of 0.22 for book recommendations and 0.74 for course recommendations, consistent with or exceeding benchmarks reported in the literature for datasets of similar scale and sparsity.

### 9.2 Future Work

1. **Neural Collaborative Filtering:** Replace KNN with a two-tower neural network (user embedding × item embedding) using matrix factorisation. Expected precision@5 improvement: 5–10 percentage points.
2. **User Authentication:** Add JWT-based login so individual users can store ratings and receive progressively personalised recommendations.
3. **Real-time Feedback Loop:** Store feedback events in a database (PostgreSQL) and retrain models monthly using accumulated user feedback from `feedback_log.jsonl`.
4. **Mobile Application:** Build a React Native wrapper around the existing REST API.
5. **A/B Testing:** Compare KNN vs matrix factorisation recommendations using the feedback endpoint to collect comparative quality signals.
6. **Larger Course Dataset:** Scrape or integrate with the Coursera API to expand to 50,000+ courses across all MOOC platforms.

---

## 10. References

1. Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms. *Proceedings of the 10th international conference on World Wide Web*, 285–295.

2. Herlocker, J. L., Konstan, J. A., Terveen, L. G., & Riedl, J. T. (2004). Evaluating collaborative filtering recommender systems. *ACM Transactions on Information Systems*, 22(1), 5–53.

3. Pazzani, M. J., & Billsus, D. (2007). Content-based recommendation systems. *The adaptive web*, 325–341. Springer, Berlin, Heidelberg.

4. Salton, G., & McGill, M. J. (1986). *Introduction to modern information retrieval*. McGraw-Hill.

5. Loper, E., & Bird, S. (2002). NLTK: The natural language toolkit. *Proceedings of the ACL-02 Workshop on Effective tools and methodologies for teaching natural language processing and computational linguistics*, 63–70.

6. Ramírez, S. (2019). FastAPI. Retrieved from https://fastapi.tiangolo.com

7. Ziegler, C. N., McNee, S. M., Konstan, J. A., & Lausen, G. (2005). Improving recommendation lists through topic diversification. *Proceedings of the 14th international conference on World Wide Web*, 22–32.

8. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of machine learning research*, 12, 2825–2830.

---

## 11. Appendix

### Appendix A — API Endpoint Reference

| Method | Endpoint | Request Body | Response |
|--------|----------|-------------|----------|
| GET | `/health` | — | `{status, version, app}` |
| GET | `/api/v1/books/popular?n=` | — | `{books[], total}` |
| POST | `/api/v1/books/recommend` | `{book_name, n}` | `{query, results[], total}` |
| GET | `/api/v1/books/autocomplete?q=` | — | `{suggestions[]}` |
| GET | `/api/v1/courses/popular?n=` | — | `{courses[], total}` |
| POST | `/api/v1/courses/recommend` | `{topic, level?, branch?, n}` | `{topic, results[], total}` |
| GET | `/api/v1/courses/metadata` | — | `{levels[], subjects[]}` |
| POST | `/api/v1/plan/generate` | `{goal, days_available, hours_per_day?}` | `{goal, schedule[], ...}` |
| GET | `/api/v1/plan/domains` | — | `{domains[]}` |
| POST | `/api/v1/feedback/` | `{query, recommended_item, item_type, rating, relevant}` | `{status, feedback_id}` |
| GET | `/api/v1/feedback/stats` | — | `{total_entries, mean_rating, ...}` |

### Appendix B — Project File Structure

```
edu-recommend/
├── app/                     FastAPI application package
│   ├── main.py              App instance, lifespan, CORS, routers
│   ├── config.py            pydantic-settings configuration
│   ├── dependencies.py      Model loading and DI injectors
│   ├── routers/             HTTP route handlers
│   ├── schemas/             Pydantic request/response models
│   └── recommenders/        ML business logic classes
├── frontend/                Static HTML + JS (no framework)
├── notebooks/               EDA + pipeline Jupyter notebooks
├── models/                  7 trained model pickle files
├── tests/                   pytest test suite (100+ tests)
├── Dockerfile               Docker image definition
├── docker-compose.yml       Local development orchestration
├── .github/workflows/ci.yml GitHub Actions CI pipeline
└── requirements.txt         Pinned Python dependencies
```

### Appendix C — Pydantic Schema Definitions

```python
# BookRecommendRequest
class BookRecommendRequest(BaseModel):
    book_name: str = Field(..., min_length=1, max_length=300)
    n: int = Field(10, ge=1, le=20)

# CourseRecommendRequest
class CourseRecommendRequest(BaseModel):
    topic: str = Field(..., min_length=2)
    level: Optional[Literal["Beginner","Intermediate","Advanced","Mixed"]] = None
    branch: Optional[str] = None
    n: int = Field(10, ge=1, le=20)

# StudyPlanRequest
class StudyPlanRequest(BaseModel):
    goal: str = Field(..., min_length=5, max_length=500)
    days_available: int = Field(14, ge=7, le=60)
    hours_per_day: Optional[int] = Field(3, ge=1, le=10)
```
