"""
app/main.py  — UPDATED Day 10
FastAPI application entry point with custom error handlers, startup logs, /api/v1/info.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.config import settings
from app.dependencies import load_models
from app.routers import books, courses, plan, feedback


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"\n{'='*55}")
    print(f"  {settings.app_name} v{settings.app_version}")
    print(f"  Loading ML models ...")
    print(f"{'='*55}")
    load_models()
    print(f"{'='*55}")
    print(f"  Ready  →  http://0.0.0.0:8000")
    print(f"  Docs   →  http://0.0.0.0:8000/docs")
    print(f"{'='*55}\n")
    yield
    print(f"\n[{settings.app_name}] Shutting down.")


app = FastAPI(
    title       = settings.app_name,
    version     = settings.app_version,
    description = settings.app_description,
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom 422 — returns clean field-level error messages."""
    errors = []
    for error in exc.errors():
        field   = " -> ".join(str(loc) for loc in error["loc"] if loc != "body")
        message = error["msg"]
        errors.append({"field": field, "message": message})
    return JSONResponse(status_code=422, content={"error": "Validation failed", "fields": errors})


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "path": str(request.url.path), "hint": "Check /docs"},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(books.router)
app.include_router(courses.router)
app.include_router(plan.router)
app.include_router(feedback.router)


@app.get("/health", tags=["system"], summary="Health check")
async def health():
    return JSONResponse(content={"status": "ok", "version": settings.app_version, "app": settings.app_name})


@app.get("/api/v1/info", tags=["system"], summary="API endpoint summary")
async def api_info():
    return JSONResponse(content={
        "app": settings.app_name, "version": settings.app_version,
        "endpoints": {
            "books"   : ["GET /api/v1/books/popular", "POST /api/v1/books/recommend", "GET /api/v1/books/autocomplete"],
            "courses" : ["GET /api/v1/courses/popular", "POST /api/v1/courses/recommend", "GET /api/v1/courses/autocomplete", "GET /api/v1/courses/metadata"],
            "plan"    : ["POST /api/v1/plan/generate", "GET /api/v1/plan/domains"],
            "feedback": ["POST /api/v1/feedback/", "GET /api/v1/feedback/stats"],
            "system"  : ["GET /health", "GET /api/v1/info", "GET /docs"],
        },
    })


try:
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
except RuntimeError:
    pass
