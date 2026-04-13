"""
tests/test_api.py
==================
FastAPI endpoint tests using TestClient.
All plan tests work without ML pickles (conftest.py initialises StudyPlanGenerator).
Book/course ML tests are skipped when pickles are absent (SKIP_MODEL_TESTS=1 or no pickles).
"""

import os
import pytest
from fastapi.testclient import TestClient

SKIP_ML = os.getenv("SKIP_MODEL_TESTS", "0") == "1" or \
          not os.path.exists("models/book_model.pkl")
skip_ml = pytest.mark.skipif(SKIP_ML, reason="ML pickles not in models/")

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


# ── System ────────────────────────────────────────────────────────────────────

class TestSystem:

    def test_health_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_body(self):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_api_info_200(self):
        assert client.get("/api/v1/info").status_code == 200

    def test_api_info_has_groups(self):
        data = client.get("/api/v1/info").json()
        for g in ["books", "courses", "plan", "feedback"]:
            assert g in data["endpoints"]

    def test_docs_200(self):
        assert client.get("/docs").status_code == 200

    def test_unknown_route_404(self):
        assert client.get("/api/v1/does-not-exist-xyz").status_code == 404


# ── Books ─────────────────────────────────────────────────────────────────────

class TestBooks:

    @skip_ml
    def test_popular_200(self):
        assert client.get("/api/v1/books/popular").status_code == 200

    @skip_ml
    def test_popular_has_books(self):
        data = client.get("/api/v1/books/popular?n=10").json()
        assert "books" in data
        assert "total" in data
        assert len(data["books"]) > 0

    @skip_ml
    def test_recommend_known_book(self):
        from app.dependencies import get_book_rec
        rec = get_book_rec()
        if not rec:
            pytest.skip("BookRecommender not loaded")
        title = rec.all_titles[0]
        r = client.post("/api/v1/books/recommend",
                        json={"book_name": title, "n": 5})
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert len(data["results"]) > 0

    def test_recommend_unknown_book_404(self):
        r = client.post("/api/v1/books/recommend",
                        json={"book_name": "XYZZY_NOTABOOK_99999", "n": 5})
        assert r.status_code in (404, 503)

    def test_recommend_invalid_n_422(self):
        r = client.post("/api/v1/books/recommend",
                        json={"book_name": "Harry Potter", "n": -1})
        assert r.status_code == 422

    def test_recommend_n_too_large_422(self):
        r = client.post("/api/v1/books/recommend",
                        json={"book_name": "Harry Potter", "n": 999})
        assert r.status_code == 422

    def test_recommend_empty_name_422(self):
        r = client.post("/api/v1/books/recommend",
                        json={"book_name": "", "n": 5})
        assert r.status_code == 422

    def test_422_clean_structure(self):
        r = client.post("/api/v1/books/recommend",
                        json={"book_name": "Harry Potter", "n": -1})
        data = r.json()
        assert r.status_code == 422
        assert "error" in data
        assert "fields" in data
        assert isinstance(data["fields"], list)

    @skip_ml
    def test_autocomplete_returns_list(self):
        r = client.get("/api/v1/books/autocomplete?q=harry")
        assert r.status_code == 200
        assert "suggestions" in r.json()

    def test_autocomplete_short_q_422(self):
        r = client.get("/api/v1/books/autocomplete?q=a")
        assert r.status_code == 422


# ── Courses ───────────────────────────────────────────────────────────────────

class TestCourses:

    @skip_ml
    def test_popular_200(self):
        assert client.get("/api/v1/courses/popular").status_code == 200

    @skip_ml
    def test_recommend_basic(self):
        r = client.post("/api/v1/courses/recommend",
                        json={"topic": "machine learning", "n": 5})
        assert r.status_code == 200
        assert len(r.json()["results"]) > 0

    def test_recommend_invalid_level_422(self):
        r = client.post("/api/v1/courses/recommend",
                        json={"topic": "python", "level": "SuperExpert"})
        assert r.status_code == 422

    def test_recommend_short_topic_422(self):
        r = client.post("/api/v1/courses/recommend", json={"topic": "a"})
        assert r.status_code == 422

    @skip_ml
    def test_metadata_200(self):
        r = client.get("/api/v1/courses/metadata")
        assert r.status_code == 200
        assert "levels" in r.json()


# ── Plan ──────────────────────────────────────────────────────────────────────

class TestPlan:
    """All plan tests work without ML pickles — conftest initialises StudyPlanGenerator."""

    def test_generate_200(self):
        r = client.post("/api/v1/plan/generate",
                        json={"goal": "prepare for GATE exam", "days_available": 14})
        assert r.status_code == 200

    def test_generate_response_structure(self):
        r = client.post("/api/v1/plan/generate",
                        json={"goal": "learn machine learning", "days_available": 7})
        data = r.json()
        for key in ["goal", "detected_keywords", "detected_domain",
                    "total_days", "schedule"]:
            assert key in data, f"Missing key: {key}"

    def test_generate_correct_day_count(self):
        days = 21
        r    = client.post("/api/v1/plan/generate",
                           json={"goal": "crack Amazon interview",
                                 "days_available": days})
        assert r.status_code == 200
        data = r.json()
        assert data["total_days"] == days
        assert len(data["schedule"]) == days

    def test_generate_each_day_has_required_fields(self):
        r        = client.post("/api/v1/plan/generate",
                               json={"goal": "DSA practice", "days_available": 7})
        schedule = r.json()["schedule"]
        required = {"day", "topic", "subtopics", "hours", "day_type"}
        for day in schedule:
            missing = required - set(day.keys())
            assert not missing, f"Day {day.get('day')} missing: {missing}"

    def test_generate_last_day_is_revision(self):
        r        = client.post("/api/v1/plan/generate",
                               json={"goal": "prepare for GATE", "days_available": 14})
        schedule = r.json()["schedule"]
        assert schedule[-1]["day_type"] == "revision"

    def test_generate_min_7_days(self):
        r = client.post("/api/v1/plan/generate",
                        json={"goal": "learn python", "days_available": 7})
        assert r.status_code == 200
        assert len(r.json()["schedule"]) == 7

    def test_generate_max_60_days(self):
        r = client.post("/api/v1/plan/generate",
                        json={"goal": "crack GATE exam", "days_available": 60})
        assert r.status_code == 200
        assert len(r.json()["schedule"]) == 60

    def test_generate_days_below_7_returns_422(self):
        r = client.post("/api/v1/plan/generate",
                        json={"goal": "study", "days_available": 3})
        assert r.status_code == 422

    def test_generate_days_above_60_returns_422(self):
        r = client.post("/api/v1/plan/generate",
                        json={"goal": "study", "days_available": 999})
        assert r.status_code == 422

    def test_generate_goal_too_short_returns_422(self):
        r = client.post("/api/v1/plan/generate",
                        json={"goal": "hi", "days_available": 14})
        assert r.status_code == 422

    def test_generate_unknown_goal_default_domain(self):
        r    = client.post("/api/v1/plan/generate",
                           json={"goal": "learn to bake sourdough bread",
                                 "days_available": 7})
        assert r.status_code == 200
        assert r.json()["detected_domain"] == "default"

    def test_generate_extra_fields_ignored(self):
        r = client.post("/api/v1/plan/generate",
                        json={"goal": "prepare for GATE exam",
                              "days_available": 14,
                              "unknown_extra": "this_should_be_ignored"})
        assert r.status_code == 200

    def test_domains_endpoint_200(self):
        assert client.get("/api/v1/plan/domains").status_code == 200

    def test_domains_has_items(self):
        data = client.get("/api/v1/plan/domains").json()
        assert "domains" in data
        assert len(data["domains"]) >= 5


# ── Feedback ──────────────────────────────────────────────────────────────────

class TestFeedback:

    def test_submit_book_feedback_200(self):
        r = client.post("/api/v1/feedback/", json={
            "query": "machine learning",
            "recommended_item": "Python for Data Science",
            "item_type": "book",
            "rating": 5,
            "relevant": True,
        })
        assert r.status_code == 200

    def test_submit_course_feedback_200(self):
        r = client.post("/api/v1/feedback/", json={
            "query": "deep learning",
            "recommended_item": "Neural Networks Course",
            "item_type": "course",
            "rating": 4,
            "relevant": True,
        })
        assert r.status_code == 200

    def test_feedback_response_has_status(self):
        r = client.post("/api/v1/feedback/", json={
            "query": "test", "recommended_item": "test item",
            "item_type": "book", "rating": 3, "relevant": False,
        })
        data = r.json()
        assert data["status"] == "recorded"
        assert "feedback_id" in data

    def test_feedback_invalid_rating_422(self):
        r = client.post("/api/v1/feedback/", json={
            "query": "test", "recommended_item": "item",
            "item_type": "book", "rating": 10, "relevant": True,
        })
        assert r.status_code == 422

    def test_feedback_invalid_item_type_422(self):
        r = client.post("/api/v1/feedback/", json={
            "query": "test", "recommended_item": "item",
            "item_type": "magazine", "rating": 4, "relevant": True,
        })
        assert r.status_code == 422

    def test_feedback_stats_200(self):
        assert client.get("/api/v1/feedback/stats").status_code == 200
