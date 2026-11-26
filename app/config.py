"""
app/config.py
=============
Central configuration using pydantic-settings.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_dir: str = "models"
    max_recommendations: int = 20
    min_recommendations: int = 1
    feedback_log_path: str = "feedback_log.jsonl"
    debug: bool = False

    app_name: str = "EduRecommend"
    app_version: str = "2.0.0"
    app_description: str = (
        "Personalised book, course, and study plan recommender. "
        "Book recommendations use KNN collaborative filtering on 270k+ ratings. "
        "Course recommendations use TF-IDF content-based filtering. "
        "Study plans use NLTK keyword extraction."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def resolved_model_dir(self) -> str:
        """Resolve model_dir relative to project root if relative path is missing in CWD."""
        p = Path(self.model_dir)
        if p.is_absolute() or p.exists():
            return str(p)
        root_rel = Path(__file__).resolve().parent.parent / self.model_dir
        return str(root_rel) if root_rel.exists() else str(p)

    def resolved_feedback_log_path(self) -> str:
        """Use /tmp when running on read-only serverless filesystems like Vercel."""
        p = Path(self.feedback_log_path)
        if p.is_absolute():
            return str(p)
        if os.environ.get("VERCEL") or not os.access(Path.cwd(), os.W_OK):
            return str(Path("/tmp") / p.name)
        return str(p)


settings = Settings()
