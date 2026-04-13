"""
app/routers/plan.py
====================
FastAPI router for the study plan generator endpoint.

Prefix  : /api/v1/plan
Tag     : study-plan

Endpoints
---------
POST /generate   → NLTK keyword extraction → day-by-day schedule JSON
GET  /domains    → list of supported domains and their topic counts
"""

from fastapi import APIRouter, Depends

from app.dependencies import get_plan_gen
from app.recommenders.study_plan import StudyPlanGenerator, TOPIC_MAP
from app.schemas.plan import StudyPlanRequest, StudyPlanResponse, DayPlan

router = APIRouter(
    prefix="/api/v1/plan",
    tags=["study-plan"],
)


@router.post(
    "/generate",
    response_model=StudyPlanResponse,
    summary="Generate a personalised study plan",
    description=(
        "Accepts a free-text goal and number of available days. "
        "Uses NLTK keyword extraction to detect the study domain, "
        "then generates a day-by-day schedule with revision days every 7 days "
        "and a mock test on the final day. "
        "Supported domains: GATE, DSA, Web Dev, ML, System Design, "
        "Amazon/Interviews, Python, Cloud, Data Science, Competitive Programming."
    ),
)
async def generate_plan(
    req: StudyPlanRequest,
    gen: StudyPlanGenerator = Depends(get_plan_gen),
):
    raw = gen.generate(
        goal           = req.goal,
        days_available = req.days_available,
        hours_per_day  = req.hours_per_day or 3,
    )

    schedule = [
        DayPlan(
            day      = d["day"],
            topic    = d["topic"],
            subtopics= d["subtopics"],
            hours    = d["hours"],
            day_type = d["day_type"],
        )
        for d in raw["schedule"]
    ]

    return StudyPlanResponse(
        goal               = raw["goal"],
        detected_keywords  = raw["detected_keywords"],
        detected_domain    = raw["detected_domain"],
        total_days         = raw["total_days"],
        total_topics       = raw["total_topics"],
        total_hours        = raw["total_hours"],
        schedule           = schedule,
    )


@router.get(
    "/domains",
    summary="List supported study domains",
    description="Returns all supported study domains with topic counts. Useful for the frontend UI.",
)
async def get_domains():
    return {
        "domains": [
            {
                "name"        : domain,
                "topic_count" : len(topics),
                "sample_topics": topics[:3],
            }
            for domain, topics in TOPIC_MAP.items()
            if domain != "default"
        ]
    }
