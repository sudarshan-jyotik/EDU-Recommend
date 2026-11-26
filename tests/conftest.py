"""
tests/conftest.py
==================
Pytest configuration — runs before any test file.

Initialises the StudyPlanGenerator (which needs no pickles) so that
plan-related tests work without ML model files.

ML-dependent tests (books/courses recommenders) are guarded with
skip_ml = pytest.mark.skipif(SKIP_ML, ...) in each test file.
"""

import os
import pytest

# Set this env var to skip all tests that need ML artifacts (for CI without models)
HAS_BOOK_MODEL = os.path.exists("models/books_bundle.npz") or os.path.exists("models/book_model.pkl")
SKIP_ML = os.getenv("SKIP_MODEL_TESTS", "0") == "1" or not HAS_BOOK_MODEL


def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line(
        "markers", "skip_ml: skip if ML model pickles are missing"
    )


@pytest.fixture(scope="session", autouse=True)
def initialise_plan_generator():
    """
    Initialise StudyPlanGenerator once before ALL tests.
    Injected into app.dependencies so the /plan endpoints work in tests
    without needing any pickle files.
    """
    import app.dependencies as dep
    from app.recommenders.study_plan import StudyPlanGenerator

    if dep._plan_gen is None:
        dep._plan_gen = StudyPlanGenerator()

    yield

    # Teardown — nothing to clean up
