"""
app/dependencies.py — updated with StudyPlanGenerator
"""
from app.config import settings
from app.recommenders.book_recommender import BookRecommender
from app.recommenders.course_recommender import CourseRecommender
from app.recommenders.study_plan import StudyPlanGenerator

_book_rec: BookRecommender = None
_course_rec: CourseRecommender = None
_plan_gen: StudyPlanGenerator = None

def load_models() -> None:
    global _book_rec, _course_rec, _plan_gen
    print(f"[dependencies] Loading models from '{settings.model_dir}' ...")
    _book_rec   = BookRecommender(settings.model_dir)
    _course_rec = CourseRecommender(settings.model_dir)
    _plan_gen   = StudyPlanGenerator()
    print("[dependencies] All models loaded and ready.")

def get_book_rec() -> BookRecommender:
    return _book_rec

def get_course_rec() -> CourseRecommender:
    return _course_rec

def get_plan_gen() -> StudyPlanGenerator:
    return _plan_gen
