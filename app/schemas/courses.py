"""app/schemas/courses.py — Pydantic v2 schemas for courses endpoints."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal


class CourseResult(BaseModel):
    course_title: str
    subject: str
    level: str
    rating: float = Field(..., ge=0.0, le=5.0)
    course_url: str = ""
    score: float


class CourseRecommendRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "topic": "machine learning",
                "level": "Intermediate",
                "branch": "CSE",
                "n": 10,
            }
        }
    )
    topic: str = Field(..., min_length=2, max_length=200)
    level: Optional[Literal["Beginner", "Intermediate", "Advanced", "Mixed"]] = None
    branch: Optional[str] = None
    n: int = Field(default=10, ge=1, le=20)


class CourseRecommendResponse(BaseModel):
    topic: str
    level_filter: Optional[str] = None
    branch_boost: bool = False
    results: list[CourseResult]
    total: int


class PopularCoursesResponse(BaseModel):
    courses: list[dict]
    total: int
