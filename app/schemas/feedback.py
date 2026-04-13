"""app/schemas/feedback.py — Pydantic v2 schemas for feedback endpoints."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "machine learning",
                "recommended_item": "Python for Data Science",
                "item_type": "course",
                "rating": 5,
                "relevant": True,
            }
        }
    )
    query: str = Field(..., min_length=1, max_length=300)
    recommended_item: str = Field(..., min_length=1, max_length=300)
    item_type: Literal["book", "course"]
    rating: int = Field(..., ge=1, le=5)
    relevant: bool
    user_branch: Optional[str] = None
    user_semester: Optional[int] = Field(default=None, ge=1, le=8)


class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: str
