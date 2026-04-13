"""app/routers/books.py — Book recommendation endpoints."""

from fastapi import APIRouter, HTTPException, Depends, Query
from app.dependencies import get_book_rec
from app.schemas.books import (
    BookRecommendRequest, BookRecommendResponse,
    BookResult, PopularBooksResponse, AutocompleteResponse,
)

router = APIRouter(prefix="/api/v1/books", tags=["books"])


@router.get("/popular", response_model=PopularBooksResponse)
async def get_popular_books(
    n: int = Query(default=50, ge=1, le=200),
    rec=Depends(get_book_rec),
):
    if rec is None:
        raise HTTPException(503, detail="Book model not loaded. Run the pipeline notebook first.")
    books = rec.get_popular(n=n)
    return PopularBooksResponse(books=books, total=len(books))


@router.post("/recommend", response_model=BookRecommendResponse)
async def recommend_books(
    req: BookRecommendRequest,
    rec=Depends(get_book_rec),
):
    if rec is None:
        raise HTTPException(503, detail="Book model not loaded. Run the pipeline notebook first.")

    results_raw = rec.recommend(req.book_name, n=req.n)

    if not results_raw:
        suggestions = rec.autocomplete(req.book_name[:15], limit=6) if rec else []
        raise HTTPException(
            status_code=404,
            detail={
                "message": f'"{req.book_name}" was not found in the recommendation model.',
                "hint": "Only books with 50+ ratings are included.",
                "suggestions": suggestions,
            },
        )

    results = [
        BookResult(
            title=r["title"], author=r["author"],
            cover=r["cover"], similarity=r["similarity"],
        )
        for r in results_raw
    ]
    return BookRecommendResponse(query=req.book_name, results=results, total=len(results))


@router.get("/autocomplete", response_model=AutocompleteResponse)
async def autocomplete(
    q: str = Query(..., min_length=2, max_length=100),
    rec=Depends(get_book_rec),
):
    if rec is None:
        return AutocompleteResponse(suggestions=[])
    return AutocompleteResponse(suggestions=rec.autocomplete(q, limit=10))
