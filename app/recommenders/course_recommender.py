"""
app/recommenders/course_recommender.py  — VERSION 2
Improvements: rating-weighted re-ranking (65% cos + 35% rating), synonym expansion,
multi-query averaging, richer CS-pattern, auto-fallback on empty level filter,
and portable loading (.json.gz bundle, .pkl fallback, or raw CSV fallback).
"""

import gzip
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_SYNONYMS = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "ds": "data science",
    "dsa": "data structures algorithms",
    "os": "operating systems",
    "dbms": "database management",
    "cn": "computer networks",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "sql": "sql database",
    "aws": "amazon web services cloud",
    "k8s": "kubernetes",
}


class CourseRecommender:
    """TF-IDF content-based course recommender with rating re-ranking and CSE branch boost."""

    _CS_PATTERN = (
        r"computer|data|machine|python|programming|software|algorithm|"
        r"network|artificial|deep learning|neural|database|web|cloud|"
        r"cybersecurity|javascript|java|react|angular|django|tensorflow|"
        r"pytorch|sql|nosql|devops|docker|kubernetes"
    )

    def __init__(self, model_dir: str = "models"):
        bundle_path = os.path.join(model_dir, "courses_bundle.json.gz")
        df_pkl = os.path.join(model_dir, "courses_df.pkl")
        tfidf_pkl = os.path.join(model_dir, "course_tfidf.pkl")

        self.vectorizer = None
        if os.path.exists(bundle_path):
            with gzip.open(bundle_path, "rt", encoding="utf-8") as f:
                records = json.load(f)
            self.df = pd.DataFrame(records)
        elif os.path.exists(df_pkl):
            with open(df_pkl, "rb") as f:
                self.df = pickle.load(f)
            if os.path.exists(tfidf_pkl):
                try:
                    with open(tfidf_pkl, "rb") as f:
                        self.vectorizer = pickle.load(f)
                except Exception:
                    self.vectorizer = None
        else:
            csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "coursera_courses.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing course model files in {model_dir} and {csv_path}")
            self.df = self._df_from_csv(csv_path)

        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 3),
                sublinear_tf=True,
                min_df=2,
                max_df=0.85,
                analyzer="word",
                token_pattern=r"\b[a-z][a-z0-9]{1,}\b",
            )
            self.tf_matrix = self.vectorizer.fit_transform(self.df["combined_text"])
        else:
            self.tf_matrix = self.vectorizer.transform(self.df["combined_text"])

        ratings = pd.to_numeric(self.df["rating"], errors="coerce").fillna(4.0)
        r_min, r_max = ratings.min(), ratings.max()
        self.norm_ratings = ((ratings - r_min) / (r_max - r_min + 1e-9)).values
        self.all_titles = self.df["course_title"].tolist()
        print(f"[CourseRecommender v2] {len(self.df)} courses  vocab={len(self.vectorizer.vocabulary_):,}")

    @staticmethod
    def _df_from_csv(csv_path: Path) -> pd.DataFrame:
        from scripts.build_models import _extract_clean_subject

        df = pd.read_csv(csv_path)
        col_map = {}
        for c in ["Course Name", "course_title", "name", "title"]:
            if c in df.columns:
                col_map[c] = "course_title"
                break
        for c in ["Skills", "skills", "subject", "Subject", "category"]:
            if c in df.columns:
                col_map[c] = "subject_raw"
                break
        for c in ["Course Description", "description", "Description"]:
            if c in df.columns:
                col_map[c] = "description"
                break
        for c in ["Difficulty Level", "level", "difficulty"]:
            if c in df.columns:
                col_map[c] = "level"
                break
        for c in ["Course Rating", "rating", "Rating"]:
            if c in df.columns:
                col_map[c] = "rating"
                break
        for c in ["Course URL", "course_url", "url"]:
            if c in df.columns:
                col_map[c] = "course_url"
                break

        df = df.rename(columns=col_map).dropna(subset=["course_title"]).copy()
        for col, default in [("subject_raw", "General"), ("description", ""), ("level", "Mixed"), ("rating", 4.0), ("course_url", "")]:
            if col not in df.columns:
                df[col] = default
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(4.0).clip(0.0, 5.0)
        level_map = {"beginner": "Beginner", "intermediate": "Intermediate", "advanced": "Advanced", "mixed": "Mixed", "all levels": "Mixed", "all": "Mixed", "expert": "Advanced"}
        df["level"] = df["level"].astype(str).str.strip().str.lower().map(level_map).fillna("Mixed")
        df = df.drop_duplicates(subset=["course_title"]).reset_index(drop=True)
        df["subject"] = [_extract_clean_subject(s, d) for s, d in zip(df["subject_raw"], df["description"])]
        clean = lambda s: re.sub(r"[^a-z0-9 ]", " ", str(s).lower().strip())
        df["combined_text"] = df.apply(
            lambda r: " ".join(filter(None, [clean(r["course_title"])] * 3 + [clean(r["subject_raw"])] * 2 + [clean(r["level"]), clean(str(r["description"])[:300])])),
            axis=1,
        )
        return df[["course_title", "subject", "level", "rating", "course_url", "combined_text"]]

    def recommend(self, topic: str, level: str = None, branch: str = None, n: int = 10) -> list[dict]:
        expanded = self._expand(topic)
        scores = self._get_scores(topic)
        if expanded != topic.lower().strip():
            scores = (scores + self._get_scores(expanded)) / 2.0

        result_df = self.df.copy()
        result_df["_cos"] = scores
        # Attach norm_ratings BEFORE filtering by level so row indices stay aligned
        result_df["_norm_rating"] = self.norm_ratings

        if branch and branch.upper() == "CSE":
            cs_mask = (
                result_df["subject"].str.contains(self._CS_PATTERN, case=False, na=False)
                | result_df["course_title"].str.contains(self._CS_PATTERN, case=False, na=False)
            )
            result_df.loc[cs_mask, "_cos"] *= 1.2

        if level:
            filtered = result_df[result_df["level"].str.lower() == level.lower()]
            result_df = filtered if len(filtered) >= 3 else result_df

        result_df["score"] = 0.65 * result_df["_cos"] + 0.35 * result_df["_norm_rating"]
        top = result_df.nlargest(n, "score")
        return [
            {
                "course_title": row["course_title"],
                "subject": row["subject"],
                "level": row["level"],
                "rating": round(float(row["rating"]), 2),
                "course_url": row.get("course_url", ""),
                "score": round(float(row["score"]), 4),
            }
            for _, row in top.iterrows()
        ]

    def get_popular(self, n: int = 30) -> list[dict]:
        top = self.df.nlargest(n, "rating")
        return [
            {
                "course_title": row["course_title"],
                "subject": row["subject"],
                "level": row["level"],
                "rating": round(float(row["rating"]), 2),
                "course_url": row.get("course_url", ""),
            }
            for _, row in top.iterrows()
        ]

    def autocomplete(self, q: str, limit: int = 8) -> list[str]:
        if not q or len(q.strip()) < 2:
            return []
        ql = q.strip().lower()
        prefix = [t for t in self.all_titles if t.lower().startswith(ql)]
        substr = [t for t in self.all_titles if ql in t.lower() and not t.lower().startswith(ql)]
        return (prefix + substr)[:limit]

    def get_levels(self) -> list[str]:
        return sorted(self.df["level"].dropna().unique().tolist())

    def get_subjects(self) -> list[str]:
        return sorted(self.df["subject"].dropna().unique().tolist())

    def _expand(self, topic: str) -> str:
        words = topic.lower().strip().split()
        return " ".join(_SYNONYMS.get(w, w) for w in words)

    def _get_scores(self, topic: str) -> np.ndarray:
        qv = self.vectorizer.transform([topic.lower().strip()])
        return cosine_similarity(qv, self.tf_matrix).flatten()
