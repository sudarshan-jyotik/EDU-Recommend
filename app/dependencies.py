"""
app/dependencies.py
===================
Loads and caches the BookRecommender, CourseRecommender, and StudyPlanGenerator.
Supports both FastAPI lifespan startup and lazy initialization (for serverless / TestClient).
"""

import threading
from typing import Optional

from app.config import settings
from app.recommenders.book_recommender import BookRecommender
from app.recommenders.course_recommender import CourseRecommender
from app.recommenders.study_plan import StudyPlanGenerator

_book_rec: Optional[BookRecommender] = None
_course_rec: Optional[CourseRecommender] = None
_plan_gen: Optional[StudyPlanGenerator] = None
_loaded = False
_lock = threading.Lock()


def load_models(force: bool = False) -> None:
    """Load ML models safely; missing/broken artifacts degrade gracefully instead of crashing."""
    global _book_rec, _course_rec, _plan_gen, _loaded
    with _lock:
        if _loaded and not force:
            return
        model_dir = settings.resolved_model_dir()
        print(f"[dependencies] Loading models from '{model_dir}' ...")

        try:
            _book_rec = BookRecommender(model_dir)
        except FileNotFoundError as exc:
            _book_rec = None
            print(f"[dependencies] Warning: BookRecommender unavailable ({exc})")
        except Exception as exc:
            _book_rec = None
            print(f"[dependencies] Warning: BookRecommender failed to load ({exc})")

        try:
            _course_rec = CourseRecommender(model_dir)
        except FileNotFoundError as exc:
            _course_rec = None
            print(f"[dependencies] Warning: CourseRecommender unavailable ({exc})")
        except Exception as exc:
            _course_rec = None
            print(f"[dependencies] Warning: CourseRecommender failed to load ({exc})")

        if _plan_gen is None or force:
            _plan_gen = StudyPlanGenerator()

        _loaded = True
        print("[dependencies] Initialization complete.")


def get_book_rec() -> Optional[BookRecommender]:
    if not _loaded:
        load_models()
    return _book_rec


def get_course_rec() -> Optional[CourseRecommender]:
    if not _loaded:
        load_models()
    return _course_rec


def get_plan_gen() -> StudyPlanGenerator:
    if _plan_gen is None:
        load_models()
    return _plan_gen
