"""
scripts/build_models.py
=======================
Builds the v2 Book & Course recommender artifacts from raw CSVs into `models/`.

Produces version-independent compressed artifacts (work across any Python/NumPy/Pandas/sklearn):
  - models/books_bundle.npz       (float32 CSR sparse user-book matrix)
  - models/books_meta.json.gz     (book titles, popular_df records, filtered meta dict)
  - models/courses_bundle.json.gz (cleaned Coursera courses with weighted combined_text)

Optionally also writes classic `.pkl` files when `--pickles` is passed.

Usage:
    python scripts/build_models.py --data-dir data --model-dir models
"""

import argparse
import gzip
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def build_books(data_dir: Path, model_dir: Path, write_pickles: bool = False) -> None:
    books_path = data_dir / "Books.csv"
    ratings_path = data_dir / "Ratings.csv"
    users_path = data_dir / "Users.csv"

    if not (books_path.exists() and ratings_path.exists() and users_path.exists()):
        print(f"[build_books] Skipping — raw Book-Crossing CSVs not found in {data_dir}")
        return

    print("[build_books] Loading Book-Crossing CSVs ...")
    books = pd.read_csv(books_path, low_memory=False)
    ratings = pd.read_csv(ratings_path, low_memory=False)
    users = pd.read_csv(users_path, low_memory=False)

    books_raw = books[["Book-Title", "Book-Author", "Image-URL-M"]].drop_duplicates("Book-Title").copy()

    # Clean books
    img_cols = [c for c in books.columns if "Image-URL" in c]
    books = books.drop(columns=img_cols)
    books = books.dropna(subset=["Book-Title", "Book-Author", "Publisher"])
    books["Year-Of-Publication"] = pd.to_numeric(books["Year-Of-Publication"], errors="coerce")
    books = books[(books["Year-Of-Publication"] >= 1900) & (books["Year-Of-Publication"] <= 2025)].reset_index(drop=True)

    # Clean users & merge
    users = users[["User-ID"]].copy()
    merged = ratings.merge(books, on="ISBN", how="inner").merge(users, on="User-ID", how="inner")
    merged = merged[["ISBN", "Book-Title", "Book-Author", "User-ID", "Book-Rating"]]

    # RatingFilter v2: min_user=30, min_book=20, min_rating=3, per-user z-score norm
    df = merged[merged["Book-Rating"] >= 3].copy()
    user_counts = df["User-ID"].value_counts()
    df = df[df["User-ID"].isin(user_counts[user_counts >= 30].index)]
    book_counts = df["Book-Title"].value_counts()
    df = df[df["Book-Title"].isin(book_counts[book_counts >= 20].index)]
    df = df.groupby(["User-ID", "Book-Title"])["Book-Rating"].max().reset_index()

    user_stats = (
        df.groupby("User-ID")["Book-Rating"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "u_mean", "std": "u_std"})
    )
    df = df.merge(user_stats, on="User-ID", how="left")
    df["u_std"] = df["u_std"].fillna(1.0).replace(0.0, 1.0)
    df["Book-Rating-Norm"] = (df["Book-Rating"] - df["u_mean"]) / df["u_std"]
    r = df["Book-Rating-Norm"]
    df["Book-Rating-Norm"] = (r - r.min()) / (r.max() - r.min() + 1e-9)

    pivot = (
        df.pivot_table(index="Book-Title", columns="User-ID", values="Book-Rating-Norm", aggfunc="mean")
        .fillna(0.0)
        .astype(np.float32)
    )
    sparse = csr_matrix(pivot.values, dtype=np.float32)

    # PopularityBuilder v2 (raw 1-10 Bayesian average, min_ratings=20)
    pop = (
        merged[merged["Book-Rating"] > 0]
        .groupby("Book-Title")
        .agg(num_ratings=("Book-Rating", "count"), avg_rating=("Book-Rating", "mean"))
        .reset_index()
    )
    m = 20
    C = pop["avg_rating"].mean()
    pop = pop[pop["num_ratings"] >= m].copy()
    pop["score"] = (pop["num_ratings"] * pop["avg_rating"] + m * C) / (pop["num_ratings"] + m)
    pop = pop.sort_values("score", ascending=False).reset_index(drop=True)
    pop = pop.merge(books_raw, on="Book-Title", how="left")

    # Keep only metadata for books in pivot or top-250 popular (shrinks 30MB -> ~300KB)
    needed_titles = set(pivot.index.tolist()) | set(pop.head(250)["Book-Title"].tolist())
    slim_meta = books_raw[books_raw["Book-Title"].isin(needed_titles)].copy()

    # Save version-proof bundle
    model_dir.mkdir(parents=True, exist_ok=True)
    save_npz(model_dir / "books_bundle.npz", sparse)

    meta_map = {}
    for _, row in slim_meta.iterrows():
        title = str(row["Book-Title"])
        author = "" if pd.isna(row["Book-Author"]) else str(row["Book-Author"]).strip()
        cover = "" if pd.isna(row["Image-URL-M"]) else str(row["Image-URL-M"]).strip()
        meta_map[title] = {
            "author": author or "Unknown",
            "cover": cover if cover.startswith("http") else "",
        }

    popular_records = []
    for _, row in pop.head(250).iterrows():
        cover = "" if pd.isna(row.get("Image-URL-M")) else str(row.get("Image-URL-M")).strip()
        author = "" if pd.isna(row.get("Book-Author")) else str(row.get("Book-Author")).strip()
        popular_records.append(
            {
                "Book-Title": str(row["Book-Title"]),
                "Book-Author": author or "Unknown",
                "num_ratings": int(row["num_ratings"]),
                "avg_rating": round(float(row["avg_rating"]), 3),
                "score": round(float(row["score"]), 3),
                "Image-URL-M": cover if cover.startswith("http") else "",
                "cover": cover if cover.startswith("http") else "",
            }
        )

    bundle_meta = {
        "titles": pivot.index.tolist(),
        "popular": popular_records,
        "meta": meta_map,
    }
    with gzip.open(model_dir / "books_meta.json.gz", "wt", encoding="utf-8") as f:
        json.dump(bundle_meta, f, ensure_ascii=False)

    print(
        f"[build_books] Saved books_bundle.npz ({(model_dir / 'books_bundle.npz').stat().st_size / 1024:.1f} KB) "
        f"& books_meta.json.gz ({(model_dir / 'books_meta.json.gz').stat().st_size / 1024:.1f} KB) "
        f"| books={len(bundle_meta['titles'])}, shape={pivot.shape}"
    )

    if write_pickles:
        knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(31, pivot.shape[0]), n_jobs=-1)
        knn.fit(sparse)
        for fname, obj in [
            ("book_model.pkl", knn),
            ("book_pivot.pkl", pivot),
            ("popular_df.pkl", pop),
            ("books_meta.pkl", slim_meta),
        ]:
            with open(model_dir / fname, "wb") as f:
                pickle.dump(obj, f)
        print("[build_books] Also wrote classic book pickles.")


def _extract_clean_subject(skills_raw: str, desc_raw: str) -> str:
    """Derive a concise readable subject/skills label for UI cards."""
    text = str(skills_raw or "").strip()
    if text and text.lower() != "nan":
        # Normalise multiple spaces into separators while keeping hyphenated tags readable
        tokens = [t.strip() for t in re.split(r"\s{2,}|,", text) if t.strip()]
        if len(tokens) >= 2:
            # Drop trailing category slugs if redundant, keep first 4 concise skills
            return " · ".join(tokens[:4])[:90]
        return text[:90]
    desc = str(desc_raw or "").strip()
    return (desc[:80] + "…") if len(desc) > 80 else (desc or "General")


def build_courses(data_dir: Path, model_dir: Path, write_pickles: bool = False) -> None:
    csv_path = data_dir / "coursera_courses.csv"
    if not csv_path.exists():
        print(f"[build_courses] Skipping — {csv_path} not found")
        return

    print("[build_courses] Loading Coursera CSV ...")
    df = pd.read_csv(csv_path)

    # Map columns accurately (Skills -> subject, Course Description -> description)
    col_map = {}
    for c in ["Course Name", "course_title", "name", "title", "Course Title", "CourseName"]:
        if c in df.columns:
            col_map[c] = "course_title"
            break
    for c in ["Skills", "skills", "subject", "Subject", "category", "Category"]:
        if c in df.columns:
            col_map[c] = "subject_raw"
            break
    for c in ["Course Description", "description", "Description"]:
        if c in df.columns:
            col_map[c] = "description"
            break
    for c in ["Difficulty Level", "level", "difficulty", "Level", "DifficultyLevel"]:
        if c in df.columns:
            col_map[c] = "level"
            break
    for c in ["Course Rating", "rating", "Rating", "score", "CourseRating"]:
        if c in df.columns:
            col_map[c] = "rating"
            break
    for c in ["Course URL", "course_url", "url", "URL", "link", "CourseURL"]:
        if c in df.columns:
            col_map[c] = "course_url"
            break
    for c in ["University", "university"]:
        if c in df.columns:
            col_map[c] = "university"
            break

    df = df.rename(columns=col_map)
    for col, default in [
        ("subject_raw", "General"),
        ("description", ""),
        ("level", "Mixed"),
        ("rating", 4.0),
        ("course_url", ""),
        ("university", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    df = df.dropna(subset=["course_title"]).copy()
    df["course_title"] = df["course_title"].astype(str).str.strip()
    df["subject_raw"] = df["subject_raw"].fillna("General").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["course_url"] = df["course_url"].fillna("").astype(str)
    df["university"] = df["university"].fillna("").astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating"] = df["rating"].fillna(df["rating"].median()).clip(0.0, 5.0)

    level_map = {
        "beginner": "Beginner",
        "intermediate": "Intermediate",
        "advanced": "Advanced",
        "mixed": "Mixed",
        "all levels": "Mixed",
        "all": "Mixed",
        "expert": "Advanced",
    }
    df["level"] = df["level"].astype(str).str.strip().str.lower().map(level_map).fillna("Mixed")
    df = df.drop_duplicates(subset=["course_title"], keep="first").reset_index(drop=True)

    df["subject"] = [
        _extract_clean_subject(s, d) for s, d in zip(df["subject_raw"], df["description"])
    ]

    def _clean(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]", " ", str(s).lower().strip())

    def _make_text(row) -> str:
        title = _clean(row["course_title"])
        subj = _clean(row["subject_raw"])
        level = _clean(row["level"])
        desc = _clean(row["description"][:300])
        return " ".join(filter(None, [title, title, title, subj, subj, level, desc]))

    df["combined_text"] = df.apply(_make_text, axis=1)

    keep_cols = ["course_title", "subject", "level", "rating", "course_url", "combined_text"]
    out_df = df[keep_cols].copy()

    model_dir.mkdir(parents=True, exist_ok=True)
    records = out_df.to_dict("records")
    for r in records:
        r["rating"] = round(float(r["rating"]), 2)
    with gzip.open(model_dir / "courses_bundle.json.gz", "wt", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(
        f"[build_courses] Saved courses_bundle.json.gz "
        f"({(model_dir / 'courses_bundle.json.gz').stat().st_size / 1024:.1f} KB) | courses={len(records)}"
    )

    if write_pickles:
        vec = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=2,
            max_df=0.85,
            analyzer="word",
            token_pattern=r"\b[a-z][a-z0-9]{1,}\b",
        )
        vec.fit(out_df["combined_text"])
        with open(model_dir / "course_tfidf.pkl", "wb") as f:
            pickle.dump(vec, f)
        with open(model_dir / "courses_df.pkl", "wb") as f:
            pickle.dump(out_df, f)
        print("[build_courses] Also wrote classic course pickles (without unused course_sim.pkl).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EduRecommend v2 model artifacts.")
    parser.add_argument("--data-dir", default="data", help="Directory containing raw CSV datasets")
    parser.add_argument("--model-dir", default="models", help="Output directory for model artifacts")
    parser.add_argument("--pickles", action="store_true", help="Also emit classic .pkl files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    build_books(data_dir, model_dir, write_pickles=args.pickles)
    build_courses(data_dir, model_dir, write_pickles=args.pickles)


if __name__ == "__main__":
    main()
