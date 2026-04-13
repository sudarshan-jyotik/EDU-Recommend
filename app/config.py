"""
app/config.py
=============
Central configuration using pydantic-settings.

Reads values from the .env file at startup.
Every other module imports the `settings` singleton from here
instead of hardcoding paths or values.

Usage:
    from app.config import settings
    path = settings.model_dir
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Directory containing all 7 pickle files
    model_dir: str = "models"

    # Maximum results any recommendation endpoint will return
    max_recommendations: int = 20

    # Minimum results (Pydantic validation lower bound)
    min_recommendations: int = 1

    # Path to the JSONL feedback log file (appended to at runtime)
    feedback_log_path: str = "feedback_log.jsonl"

    # Show detailed error messages in HTTP responses (disable in production)
    debug: bool = False

    # App display name (shown in OpenAPI docs)
    app_name: str = "EduRecommend"
    app_version: str = "1.0.0"
    app_description: str = (
        "Personalised book, course, and study plan recommender. "
        "Book recommendations use KNN collaborative filtering on 270k+ ratings. "
        "Course recommendations use TF-IDF content-based filtering. "
        "Study plans use NLTK keyword extraction."
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton — import this object everywhere instead of instantiating Settings()
settings = Settings()
