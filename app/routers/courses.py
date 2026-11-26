"""app/routers/courses.py — Course recommendation endpoints."""

from fastapi import APIRouter, Depends, Query, HTTPException
from app.dependencies import get_course_rec
from app.schemas.courses import (
    CourseRecommendRequest, CourseRecommendResponse,
    CourseResult, PopularCoursesResponse,
)

router = APIRouter(prefix="/api/v1/courses", tags=["courses"])


@router.get("/popular", response_model=PopularCoursesResponse)
async def get_popular_courses(
    n: int = Query(default=30, ge=1, le=100),
    rec=Depends(get_course_rec),
):
    if rec is None:
        raise HTTPException(503, detail="Course model not loaded.")
    courses = rec.get_popular(n=n)
    return PopularCoursesResponse(courses=courses, total=len(courses))


@router.post("/recommend", response_model=CourseRecommendResponse)
async def recommend_courses(
    req: CourseRecommendRequest,
    rec=Depends(get_course_rec),
):
    if rec is None:
        raise HTTPException(503, detail="Course model not loaded.")

    results_raw = rec.recommend(
        topic=req.topic, level=req.level, branch=req.branch, n=req.n,
    )
    results = [
        CourseResult(
            course_title=r["course_title"], subject=r["subject"],
            level=r["level"], rating=r["rating"],
            course_url=r.get("course_url", ""), score=r["score"],
        )
        for r in results_raw
    ]
    return CourseRecommendResponse(
        topic=req.topic,
        level_filter=req.level,
        branch_boost=(req.branch is not None and req.branch.upper() == "CSE"),
        results=results,
        total=len(results),
    )


@router.get("/autocomplete")
async def autocomplete(
    q: str = Query(..., min_length=2, max_length=100),
    rec=Depends(get_course_rec),
):
    if rec is None:
        return {"suggestions": []}
    return {"suggestions": rec.autocomplete(q, limit=8)}


@router.get("/metadata")
async def get_metadata(rec=Depends(get_course_rec)):
    if rec is None:
        return {"levels": [], "subjects": []}
    return {"levels": rec.get_levels(), "subjects": rec.get_subjects()[:30]}
