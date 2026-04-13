"""app/schemas/plan.py — Pydantic v2 schemas for study plan endpoints."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class DayPlan(BaseModel):
    day: int = Field(..., ge=1)
    topic: str
    subtopics: list[str]
    hours: int = Field(..., ge=1, le=12)
    day_type: str = "study"


class StudyPlanRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "goal": "prepare for GATE exam",
                "days_available": 30,
                "hours_per_day": 4,
            }
        }
    )
    goal: str = Field(..., min_length=5, max_length=500)
    days_available: int = Field(default=14, ge=7, le=60)
    hours_per_day: Optional[int] = Field(default=3, ge=1, le=10)


class StudyPlanResponse(BaseModel):
    goal: str
    detected_keywords: list[str]
    detected_domain: str
    total_days: int
    total_topics: int
    total_hours: int
    schedule: list[DayPlan]
