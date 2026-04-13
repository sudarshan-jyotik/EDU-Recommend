"""
app/routers/feedback.py
========================
FastAPI router for the user feedback endpoint.

Prefix  : /api/v1/feedback
Tag     : feedback

Endpoints
---------
POST /       → record a feedback event to feedback_log.jsonl
GET  /stats  → summary statistics of collected feedback

The feedback log is a JSON Lines file. Each line is one feedback entry.
This data accumulates at runtime and is the raw material for future
model retraining and A/B testing.
"""

import json
from datetime import datetime, timezone

from fastapi import APIRouter

from app.config import settings
from app.schemas.feedback import FeedbackRequest, FeedbackResponse

router = APIRouter(
    prefix="/api/v1/feedback",
    tags=["feedback"],
)


@router.post(
    "/",
    response_model=FeedbackResponse,
    summary="Submit recommendation feedback",
    description=(
        "Records whether a recommendation was relevant to the user. "
        "Data is appended to feedback_log.jsonl on disk. "
        "Ratings: 1 = not relevant, 5 = perfectly relevant."
    ),
)
async def submit_feedback(req: FeedbackRequest):
    timestamp = datetime.now(timezone.utc).isoformat()

    entry = {
        **req.model_dump(),
        "timestamp": timestamp,
    }

    # Append one JSON line to the feedback log file
    try:
        with open(settings.feedback_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        # Don't crash the app if feedback logging fails
        print(f"[feedback] Warning: could not write to log: {e}")

    return FeedbackResponse(
        status      = "recorded",
        message     = "Thank you! Your feedback helps improve recommendations.",
        feedback_id = timestamp,
    )


@router.get(
    "/stats",
    summary="Feedback statistics",
    description=(
        "Returns summary statistics from the feedback log file. "
        "Useful for monitoring recommendation quality over time."
    ),
)
async def get_feedback_stats():
    entries = []

    try:
        with open(settings.feedback_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        return {
            "total_entries"   : 0,
            "message"         : "No feedback has been recorded yet.",
        }

    if not entries:
        return {"total_entries": 0}

    import statistics

    book_entries   = [e for e in entries if e.get("item_type") == "book"]
    course_entries = [e for e in entries if e.get("item_type") == "course"]
    all_ratings    = [e["rating"] for e in entries if "rating" in e]
    all_relevant   = [e["relevant"] for e in entries if "relevant" in e]

    return {
        "total_entries"          : len(entries),
        "book_feedback_count"    : len(book_entries),
        "course_feedback_count"  : len(course_entries),
        "mean_rating"            : round(statistics.mean(all_ratings), 2) if all_ratings else None,
        "relevance_rate"         : round(sum(all_relevant) / len(all_relevant), 3) if all_relevant else None,
        "latest_entry_timestamp" : entries[-1].get("timestamp") if entries else None,
    }
