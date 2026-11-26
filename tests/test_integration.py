"""
tests/test_integration.py
==========================
End-to-end flow tests. Plan tests work without pickles.
ML tests are skipped if pickles are absent.
"""

import os
import pytest
from fastapi.testclient import TestClient

HAS_BOOK_MODEL = os.path.exists("models/books_bundle.npz") or os.path.exists("models/book_model.pkl")
SKIP_ML = os.getenv("SKIP_MODEL_TESTS", "0") == "1" or not HAS_BOOK_MODEL
skip_ml = pytest.mark.skipif(SKIP_ML, reason="ML artifacts not in models/")

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


class TestSystemFlow:

    def test_health(self):
        assert client.get("/health").json()["status"] == "ok"

    def test_all_endpoint_groups_exist(self):
        data = client.get("/api/v1/info").json()
        for g in ["books", "courses", "plan", "feedback", "system"]:
            assert g in data["endpoints"]

    def test_docs_loads(self):
        assert client.get("/docs").status_code == 200

    def test_unknown_route_404_with_path(self):
        r    = client.get("/api/v1/unknown-xyz")
        data = r.json()
        assert r.status_code in (404, 503)
        assert "error" in data

    def test_all_api_responses_are_json(self):
        for url in ["/health", "/api/v1/info", "/api/v1/plan/domains",
                    "/api/v1/feedback/stats"]:
            ct = client.get(url).headers.get("content-type", "")
            assert "application/json" in ct, f"{url} returned {ct}"


class TestBooksFlow:

    @skip_ml
    def test_popular_books_loads(self):
        r    = client.get("/api/v1/books/popular?n=50")
        data = r.json()
        assert r.status_code == 200
        assert data["total"] > 0

    @skip_ml
    def test_search_returns_10_results(self):
        from app.dependencies import get_book_rec
        rec = get_book_rec()
        if not rec:
            pytest.skip("Recommender not loaded")
        title = rec.all_titles[0]
        r     = client.post("/api/v1/books/recommend",
                            json={"book_name": title, "n": 10})
        assert r.status_code == 200
        assert len(r.json()["results"]) == 10

    @skip_ml
    def test_similarity_scores_valid(self):
        from app.dependencies import get_book_rec
        rec = get_book_rec()
        if not rec:
            pytest.skip("Recommender not loaded")
        title   = rec.all_titles[0]
        results = client.post("/api/v1/books/recommend",
                              json={"book_name": title, "n": 5}).json()["results"]
        for item in results:
            assert 0 <= item["similarity"] <= 100

    def test_unknown_book_returns_404(self):
        r = client.post("/api/v1/books/recommend",
                        json={"book_name": "XYZZY_NOTREAL_99999", "n": 5})
        assert r.status_code in (404, 503)

    def test_book_feedback_recorded(self):
        r = client.post("/api/v1/feedback/", json={
            "query": "Harry Potter", "recommended_item": "The Hobbit",
            "item_type": "book", "rating": 5, "relevant": True,
        })
        assert r.status_code == 200
        assert r.json()["status"] == "recorded"


class TestCoursesFlow:

    @skip_ml
    def test_popular_courses_loads(self):
        data = client.get("/api/v1/courses/popular?n=30").json()
        assert len(data["courses"]) > 0

    @skip_ml
    def test_search_basic(self):
        r = client.post("/api/v1/courses/recommend",
                        json={"topic": "machine learning", "n": 10})
        assert r.status_code == 200
        assert len(r.json()["results"]) > 0

    @skip_ml
    def test_cse_boost_flag(self):
        r = client.post("/api/v1/courses/recommend",
                        json={"topic": "data science", "branch": "CSE", "n": 5})
        assert r.json()["branch_boost"] is True

    @skip_ml
    def test_non_cse_no_boost(self):
        r = client.post("/api/v1/courses/recommend",
                        json={"topic": "data science", "branch": "ME", "n": 5})
        assert r.json()["branch_boost"] is False

    def test_invalid_level_422(self):
        r = client.post("/api/v1/courses/recommend",
                        json={"topic": "python", "level": "Expert"})
        assert r.status_code == 422

    def test_course_feedback(self):
        r = client.post("/api/v1/feedback/", json={
            "query": "ml", "recommended_item": "ML by Stanford",
            "item_type": "course", "rating": 5, "relevant": True,
        })
        assert r.status_code == 200


class TestPlanFlow:
    """All plan tests work without ML pickles."""

    def test_gate_plan(self):
        r    = client.post("/api/v1/plan/generate",
                           json={"goal": "prepare for GATE exam",
                                 "days_available": 14})
        data = r.json()
        assert r.status_code == 200
        assert data["detected_domain"] == "gate"
        assert len(data["schedule"]) == 14

    def test_web_plan(self):
        r = client.post("/api/v1/plan/generate",
                        json={"goal": "get a web development internship",
                              "days_available": 21})
        assert r.json()["detected_domain"] == "web"

    def test_ml_plan(self):
        r = client.post("/api/v1/plan/generate",
                        json={"goal": "learn machine learning from scratch",
                              "days_available": 30})
        assert r.json()["detected_domain"] == "ml"

    def test_amazon_plan(self):
        r = client.post("/api/v1/plan/generate",
                        json={"goal": "crack Amazon SDE interview",
                              "days_available": 14})
        assert r.json()["detected_domain"] == "amazon"

    def test_revision_on_day_7(self):
        r        = client.post("/api/v1/plan/generate",
                               json={"goal": "prepare for GATE", "days_available": 21})
        schedule = r.json()["schedule"]
        day7     = next(d for d in schedule if d["day"] == 7)
        assert day7["day_type"] == "revision"

    def test_last_day_always_revision(self):
        for days in [7, 14, 30]:
            r        = client.post("/api/v1/plan/generate",
                                   json={"goal": "study DSA", "days_available": days})
            schedule = r.json()["schedule"]
            assert schedule[-1]["day_type"] == "revision"

    def test_custom_hours(self):
        r        = client.post("/api/v1/plan/generate",
                               json={"goal": "learn python", "days_available": 7,
                                     "hours_per_day": 6})
        schedule = r.json()["schedule"]
        study    = [d for d in schedule if d["day_type"] == "study"]
        for d in study:
            assert d["hours"] == 6

    def test_domains_list(self):
        r    = client.get("/api/v1/plan/domains")
        data = r.json()
        names = [d["name"] for d in data["domains"]]
        for expected in ["gate", "dsa", "web", "ml", "python"]:
            assert expected in names

    def test_extra_fields_ignored(self):
        r = client.post("/api/v1/plan/generate",
                        json={"goal": "prepare for GATE exam",
                              "days_available": 14,
                              "extra_unknown_field": "ignored"})
        assert r.status_code == 200


class TestValidation:

    def test_422_clean_structure(self):
        r    = client.post("/api/v1/books/recommend",
                           json={"book_name": "HP", "n": -99})
        data = r.json()
        assert r.status_code == 422
        assert "error" in data
        assert "fields" in data
        assert isinstance(data["fields"], list)
        assert "field" in data["fields"][0]
        assert "message" in data["fields"][0]

    def test_missing_required_field_422(self):
        r = client.post("/api/v1/books/recommend", json={"n": 5})
        assert r.status_code == 422

    def test_plan_goal_too_short_422(self):
        r = client.post("/api/v1/plan/generate",
                        json={"goal": "hi", "days_available": 14})
        assert r.status_code == 422

    def test_plan_days_out_of_range_422(self):
        assert client.post("/api/v1/plan/generate",
                           json={"goal": "learn python", "days_available": 3}).status_code == 422
        assert client.post("/api/v1/plan/generate",
                           json={"goal": "learn python", "days_available": 100}).status_code == 422


class TestFeedbackStats:

    def test_submit_and_stats(self):
        for entry in [
            {"query": "ml", "recommended_item": "ML Course",
             "item_type": "course", "rating": 5, "relevant": True},
            {"query": "books", "recommended_item": "Python Book",
             "item_type": "book", "rating": 4, "relevant": True},
        ]:
            assert client.post("/api/v1/feedback/", json=entry).status_code == 200

        r    = client.get("/api/v1/feedback/stats")
        data = r.json()
        assert r.status_code == 200
        assert "total_entries" in data
        assert data["total_entries"] >= 2
