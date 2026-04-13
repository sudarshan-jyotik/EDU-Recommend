"""app/schemas/books.py — Pydantic v2 schemas for books endpoints."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class BookResult(BaseModel):
    title: str
    author: str
    cover: str
    similarity: float = Field(..., ge=0.0, le=100.0)


class BookRecommendRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "book_name": "Harry Potter and the Sorcerer's Stone",
                "n": 10,
            }
        }
    )
    book_name: str = Field(..., min_length=1, max_length=300)
    n: int = Field(default=10, ge=1, le=20)


class BookRecommendResponse(BaseModel):
    query: str
    results: list[BookResult]
    total: int


class PopularBooksResponse(BaseModel):
    books: list[dict]
    total: int


class AutocompleteResponse(BaseModel):
    suggestions: list[str]
