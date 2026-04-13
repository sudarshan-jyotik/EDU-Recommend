# Resume Bullets + Interview Preparation
## Days 20–22 — EduRecommend BTP Project

---

## Resume Bullets (copy these exactly)

### Option A — Technical emphasis (for ML/backend roles)

> Built **EduRecommend**, a 3-module ML recommendation system — KNN collaborative filtering (precision@5: 0.22) on 270k+ Book-Crossing ratings, TF-IDF content-based filtering on 3.5k Coursera courses, and an NLTK study plan generator with 10 study domains. Served via a FastAPI REST API with Pydantic validation, auto-generated Swagger docs, and a user feedback loop endpoint.

> Containerised the application with Docker (`docker-compose up` one-command run), deployed on Render.com, and set up CI with GitHub Actions — `pytest` runs on every push with 100+ test cases across unit, API, and integration test suites.

### Option B — Full-stack emphasis (for SDE/web roles)

> Built and deployed **EduRecommend** — a full-stack ML web application with a FastAPI REST backend (11 endpoints, Pydantic validation, OpenAPI docs), a decoupled vanilla JS frontend (4 pages, live autocomplete, PDF export via jsPDF), and 3 ML recommendation modules trained on 270k+ data points. Containerised with Docker, deployed on Render.com.

### Option C — Short version (for limited resume space)

> **EduRecommend** — Full-stack ML recommendation system (KNN + TF-IDF + NLTK), FastAPI REST API with Swagger UI, Docker, GitHub Actions CI, deployed on Render. Precision@5: 0.22 (books), 0.74 (courses).

---

## LinkedIn Post (Day 22)

```
🎓 Just completed my BTP project — EduRecommend!

Built a 3-module ML recommendation system for engineering students:
📚 Book recommendations → KNN collaborative filtering on 270k+ ratings
📖 Course recommendations → TF-IDF content-based filtering (3,500+ Coursera courses)
📅 Study plan generator → NLTK keyword extraction → day-by-day schedule

Production highlights:
✅ FastAPI REST API with auto-generated Swagger docs
✅ Dockerised — runs in 2 commands
✅ GitHub Actions CI — green badge on every push
✅ Deployed live on Render.com

🔗 Live app: https://your-app.onrender.com
📂 GitHub: github.com/YOUR_USERNAME/edu-recommend
📊 API docs: https://your-app.onrender.com/docs

Tech: Python · FastAPI · scikit-learn · NLTK · Pydantic · Docker · GitHub Actions

#MachineLearning #FastAPI #Python #BTP #RecommenderSystems #MLEngineering
```

---

## Interview Q&A — 25 Questions with Model Answers

### Section 1 — ML Algorithms

**Q1: Why did you choose KNN over matrix factorisation for the book recommender?**

KNN was chosen because it is interpretable and works well at this scale. With 812 books in the filtered matrix, KNN runs instantly and the cosine similarity metric directly reflects reading preference overlap between books. Matrix factorisation (SVD, ALS) would likely give better precision on sparse data by learning latent factors, but KNN is transparent — I can explain exactly why two books are similar by showing which users rated both highly. For a BTP project where the evaluator needs to understand the model, interpretability matters. Future work would use neural collaborative filtering.

**Q2: What is cosine similarity and why is it better than Euclidean distance for this use case?**

Cosine similarity measures the angle between two vectors in a high-dimensional space, regardless of their magnitudes. For ratings data, this means a user who gives 8-9-10 to the same books as a user who gives 1-2-3 will have high cosine similarity — they have the same relative preferences even though their absolute scales differ. Euclidean distance would consider them very different. Since different users use different parts of the 1-10 rating scale, cosine similarity is the correct metric.

**Q3: What is TF-IDF and how does it help with course recommendations?**

TF-IDF stands for Term Frequency–Inverse Document Frequency. It converts text into numerical vectors. TF measures how often a word appears in a specific course's description. IDF penalises words that appear in many courses — common words like "course" or "learn" get low IDF weights because they appear everywhere and don't distinguish courses. The result is a vector where rare, meaningful words like "kubernetes" or "bayesian" get high weights. Cosine similarity between the query vector and all course vectors gives a relevance score. Bigrams (ngram_range=(1,2)) help because "machine learning" as a two-word phrase is far more informative than "machine" and "learning" separately.

**Q4: What is the cold-start problem and how did you handle it?**

The cold-start problem is when a new user has no rating history, so the collaborative filter cannot find similar users or recommend relevant items. I handled it with a popularity-based fallback — the homepage shows the top-50 books by Bayesian popularity score, which is useful for any new user regardless of their history. The Bayesian average score prevents a book with a single 5-star rating from dominating the popular list by weighting each book's rating against the global mean.

**Q5: What is the Bayesian average and why use it instead of mean rating?**

The Bayesian average is: score = (v×R + m×C) / (v+m), where v = number of ratings, R = mean rating, m = minimum threshold (50), C = global mean. A book with 1 rating of 10 gets a lower Bayesian score than a book with 500 ratings averaging 8. The formula pulls books with few ratings toward the global mean, preventing noise from dominating the popular list. It's the same formula used by IMDb for their Top 250 movies ranking.

**Q6: What is precision@K and how did you compute it?**

Precision@K measures what fraction of the top-K recommendations are actually relevant to the user. In my evaluation, I split ratings 80/20 by user, defined "relevant" as books the user rated ≥ 8/10, and computed: precision@5 = (number of relevant books in top-5) / 5. Averaged across 100 test users, I got 0.22. This means on average, about 1.1 of the top-5 recommended books are books the user actually liked.

**Q7: How does recall@K differ from precision@K?**

Precision@K asks "of what we recommended, how much was relevant?" Recall@K asks "of all relevant items, how many did we recommend?" They trade off against each other. High precision with low recall means we're very selective but miss many relevant items. High recall with low precision means we recommend many things but include too much irrelevant content. For a book recommendation system where the relevant set is very small (the user only rates a handful of books as ≥8), recall@K tends to be very low even for good models.

**Q8: Why filter to users with ≥50 ratings and books with ≥50 ratings?**

Without filtering, the user-book matrix is 99.98% sparse — almost all cells are empty. KNN needs enough co-ratings between books to compute meaningful similarity. If Book A and Book B have only 2 common raters, that similarity is noise. The threshold of 50 was chosen empirically — it keeps enough books (812) and users (943) for meaningful KNN while removing noise. Lowering to 20 increases book coverage but reduces precision; raising to 100 reduces noise but excludes too many books.

---

### Section 2 — FastAPI and Backend

**Q9: What is Pydantic and why did you use it in FastAPI?**

Pydantic is a data validation library that enforces types and constraints at runtime using Python type annotations. In FastAPI, every request body is defined as a Pydantic model. When a client sends a request, FastAPI feeds the raw JSON into the Pydantic model — if any field is the wrong type or violates a constraint (like `n: -1` when the minimum is 1), Pydantic raises a validation error and FastAPI returns HTTP 422 automatically, before my route handler runs. This means my ML code never receives invalid data. The response models also guarantee the shape of what I return.

**Q10: What is dependency injection and how does it work in FastAPI?**

Dependency injection is a pattern where a function declares what it needs as parameters, and a framework provides those dependencies automatically. In FastAPI, `Depends()` is the mechanism. My routers declare `rec = Depends(get_book_rec)` as a parameter. FastAPI calls `get_book_rec()` — which returns the pre-loaded `BookRecommender` singleton — and passes it to the route handler. This means I never access global variables in my route handlers, making them independently testable (I can mock `get_book_rec()` in tests to return a fake recommender).

**Q11: Why load the ML pickles at startup instead of on each request?**

Loading 7 pickle files on every request would take 2–5 seconds and make the API unusable. Loading at startup means they're loaded once when the server starts (10–30 seconds) and kept in memory for the entire process lifetime. All requests share the same in-memory models via dependency injection. The FastAPI `lifespan` context manager is the correct place to do this — it runs before any request is served.

**Q12: What is CORS and why did you add CORSMiddleware?**

CORS (Cross-Origin Resource Sharing) is a browser security policy that blocks JavaScript from calling APIs on different domains. Since the frontend is served at `localhost:8000` and the API is also at `localhost:8000`, they're on the same origin in production (no CORS issue). But during development, if someone runs the frontend on port 3000 while the API runs on 8000, the browser would block requests without CORS headers. `CORSMiddleware` with `allow_origins=["*"]` tells the browser that the API accepts requests from any origin. In production, this should be restricted to your actual domain.

**Q13: What does the feedback endpoint do and why is it important?**

The feedback endpoint (`POST /api/v1/feedback/`) accepts a user's judgement on a recommendation — the query, the recommended item, a 1–5 rating, and a boolean relevance flag. It appends a JSON line to `feedback_log.jsonl`. This serves two purposes: it enables monitoring of recommendation quality over time (are users rating results as relevant?), and it accumulates labelled data for future model retraining. Without a feedback loop, you cannot know if your deployed model is actually helping users.

---

### Section 3 — Engineering and DevOps

**Q14: What does Docker solve in this project?**

Without Docker, running this project requires: installing Python 3.11, creating a virtual environment, installing 10+ Python packages to exact versions, downloading NLTK data, ensuring the right directory structure, and having the 7 pickle files in the right place. Docker packages all of this into a single image. Anyone with Docker installed can run `docker-compose up` and get the exact same environment in 2 commands. This eliminates "works on my machine" problems and is how real engineering teams ship software.

**Q15: What is the difference between `docker build` and `docker-compose up`?**

`docker build` creates a Docker image from the Dockerfile — it's like compiling. `docker-compose up` reads `docker-compose.yml`, builds the image if needed, and starts one or more containers from it. `docker-compose.yml` also mounts volumes (`./models:/app/models`) so the pickle files on my laptop are accessible inside the container without being baked into the image. This means I can update pickles and restart without rebuilding.

**Q16: What does GitHub Actions CI do?**

GitHub Actions is a CI (Continuous Integration) system built into GitHub. My `.github/workflows/ci.yml` defines a workflow that runs on every push: it checks out the code, installs Python 3.11, installs dependencies, and runs `pytest tests/ -v`. If any test fails, the push is flagged with a red ✗. When all tests pass, the commit gets a green ✓. The badge on the README shows whether the latest commit is passing. This tells any potential employer or collaborator that the code is tested and working.

**Q17: How did you handle the case where pickle files don't exist in CI?**

In GitHub Actions, the ML pickles are not available (they're generated locally by running notebooks and are too large to commit to GitHub). I set `SKIP_MODEL_TESTS=1` as an environment variable in the CI workflow. My test files check for this variable using `pytest.mark.skipif` — if it's set, all tests that require loaded models are skipped. The CI still runs all system tests, validation tests, plan generator tests, and integration structure tests — about 40 tests that don't need pickles. This gives meaningful CI feedback without committing 80MB of pickle files.

---

### Section 4 — System Design

**Q18: How would you scale this system to handle 10,000 concurrent users?**

Currently, the app runs as a single uvicorn process. To scale:
1. Increase `--workers` in uvicorn to use all CPU cores (CPU-bound KNN queries)
2. Add a Redis cache in front of the KNN endpoint — popular book queries repeat often
3. Move from local pickle loading to a model serving layer (TorchServe or BentoML)
4. Put the FastAPI app behind an Nginx reverse proxy for load balancing
5. Use Render's auto-scaling or move to Kubernetes for dynamic instance scaling

For the current BTP project scale (demo usage), a single Render instance is sufficient.

**Q19: Why did you choose a decoupled frontend instead of Jinja2 server-side rendering?**

Jinja2 server rendering couples the backend and frontend — the Python server must know about HTML templates. A decoupled frontend means the backend is a pure REST API and the frontend is any client that speaks JSON. Benefits: (1) the API can be tested with Postman or curl independently of the UI, (2) a mobile app could consume the same API without any backend changes, (3) the Swagger UI at /docs works automatically because FastAPI only knows about JSON, and (4) it demonstrates API-first design which is standard in industry.

**Q20: How would you add user authentication to this system?**

I would use JWT (JSON Web Tokens) with FastAPI's OAuth2 password flow. The flow: user registers → server stores hashed password in PostgreSQL → user logs in → server issues a JWT with user_id and expiry → frontend stores JWT in localStorage → every API request sends the JWT in Authorization header → FastAPI verifies the JWT in a dependency before the route handler runs. This enables per-user rating storage, personalised models, and proper session management. For the BTP project, I store profile data in localStorage to demonstrate personalisation without authentication complexity.

---

### Section 5 — Project-specific

**Q21: What was the hardest bug you fixed in this project?**

The cover images were missing in book recommendations despite being in the original dataset. The bug was in the pipeline order: `BookCleaner` ran before `DataMerger` and dropped all `Image-URL-*` columns. By the time `DataMerger` tried to save `books_meta`, the image columns were already gone. The fix was adding a `books_raw` key in `DataLoader` — saved before `BookCleaner` runs — containing `Book-Title`, `Book-Author`, and `Image-URL-M`. `DataMerger` then uses `books_raw` as the source for `books_meta`. This taught me that pipeline stage ordering matters and that data needed downstream must be captured before any transformation removes it.

**Q22: What is your evaluation metric for the book recommender and why?**

I use precision@5 and hit_rate@10. Precision@5 measures average fraction of relevant books in the top-5 (0.22). Hit Rate@10 measures the fraction of users for whom at least one relevant book appears in the top-10 (0.41). I chose these because they mirror how users actually interact with recommendations — they look at the first few results and decide if the system is useful. Recall would require knowing all books the user would have liked, which is unknowable from a partial rating history.

**Q23: How would you improve the course recommender?**

Three improvements: (1) Use a larger dataset — 3,500 Coursera courses is small; integrating with the Coursera API or adding edX/Udemy data would improve coverage. (2) Use BERT sentence embeddings instead of TF-IDF — they capture semantic meaning ("neural network" and "deep learning" would have high similarity even without exact word overlap). (3) Add implicit user feedback — if a user clicks Enrol on a course, treat that as a positive signal to personalise future recommendations.

**Q24: What does the study plan generator do when it doesn't recognise the goal?**

It falls back to the "default" domain, which has 7 generic study topics: "Core Concepts & Fundamentals", "Problem Solving Practice", "Concept Deep Dive", "Applied Exercises", "Mock Test & Review", "Weak Area Revision", "Full Mock Assessment". These cycle across the available days. In practice, the keyword-to-domain mapping covers the most common engineering student goals, so fallback is rare. The unit tests verify that this fallback produces a valid, structurally correct schedule.

**Q25: Walk me through how a request flows through your system.**

When a user searches for "Harry Potter" on the books page: (1) The JS calls `GET /api/v1/books/autocomplete?q=harry`, which returns matching titles. (2) User selects a title, JS calls `POST /api/v1/books/recommend` with `{"book_name": "Harry Potter...", "n": 10}`. (3) FastAPI receives the request and runs it through Pydantic validation — checks `book_name` is a non-empty string, `n` is between 1 and 20. (4) The `books.recommend_books` route handler runs. `Depends(get_book_rec)` injects the pre-loaded `BookRecommender`. (5) `BookRecommender.recommend()` finds the book in the pivot index, runs `kneighbors()` on the CSR sparse matrix, fetches cover URL and author from `books_meta_idx` using O(1) `.loc` lookup. (6) Results are returned as a `BookRecommendResponse` Pydantic model — FastAPI serialises to JSON. (7) The browser receives the JSON, renders 10 cards with similarity percentage bars, each with thumbs up/down feedback buttons.

---

## Day 22 Checklist — Before Submission

```
CODE
[ ] All 7 pickles generated and in models/
[ ] uvicorn app.main:app --reload starts without errors
[ ] http://localhost:8000/docs shows all 11 endpoints
[ ] All 4 frontend pages load and work
[ ] PDF export works on plan.html
[ ] Tests pass: pytest tests/ -v (with pickles)
[ ] Tests pass: SKIP_MODEL_TESTS=1 pytest tests/ -v (CI simulation)

DEPLOYMENT
[ ] Pushed to GitHub (repo is public)
[ ] GitHub Actions CI shows green badge
[ ] Deployed on Render.com — live URL works
[ ] /docs accessible on live URL
[ ] All 4 frontend pages work on live URL

DOCUMENTATION
[ ] README.md has live URL badge, screenshots, 3-command quickstart
[ ] BTP report complete (all 9 sections)
[ ] Presentation slides ready (12 slides)
[ ] Speaker notes practiced — can do it in 5 minutes

RESUME
[ ] 3 resume bullets written and added to resume
[ ] LinkedIn post published with live URL
[ ] Project added to LinkedIn profile
[ ] GitHub profile has pinned repo with good description
```
