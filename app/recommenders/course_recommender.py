"""
app/recommenders/course_recommender.py
=======================================
CourseRecommender class — wraps the 3 course ML pickles.

Created on Day 6 of the EduRecommend BTP project.
Extracted from notebooks/04_courses_pipeline.ipynb.

Algorithm: TF-IDF content-based filtering with cosine similarity.
No user ratings needed — works purely on course metadata
(title + subject + level).

Supports optional:
  - level filtering (Beginner / Intermediate / Advanced)
  - branch-based hybrid boosting (CSE users get CS/ML courses ranked higher)
"""

import pickle
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class CourseRecommender:
    """
    Content-based course recommender using TF-IDF + cosine similarity.

    Supports:
      - recommend()    — top-n courses matching a topic query
      - get_popular()  — top-n courses by rating
      - autocomplete() — matching course titles for the search dropdown
    """

    # Keywords used to identify CS/ML courses for the branch boost
    _CS_PATTERN = (
        r"computer|data|machine|python|programming|software|"
        r"algorithm|network|artificial|deep learning|neural|"
        r"database|web development|cloud|cybersecurity"
    )

    def __init__(self, model_dir: str = "models"):
        """
        Load all 3 course pickles from model_dir.
        Called once at app startup via dependencies.load_models().

        Parameters
        ----------
        model_dir : str
            Directory containing the 3 course pickle files.
        """
        def _load(filename):
            path = os.path.join(model_dir, filename)
            with open(path, "rb") as f:
                return pickle.load(f)

        self.vectorizer = _load("course_tfidf.pkl")   # fitted TfidfVectorizer
        self.sim_matrix = _load("course_sim.pkl")     # ndarray (n x n)
        self.df         = _load("courses_df.pkl")     # course metadata DataFrame

        # Pre-transform all course texts for query-time similarity
        # (used as fallback if sim_matrix lookup isn't preferred)
        self.tf_matrix = self.vectorizer.transform(self.df["combined_text"])

        # List of all titles for autocomplete
        self.all_titles = self.df["course_title"].tolist()

        print(
            f"[CourseRecommender] loaded — "
            f"{len(self.df)} courses, "
            f"vocab={len(self.vectorizer.vocabulary_):,}, "
            f"sim_matrix={self.sim_matrix.shape}"
        )

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def recommend(
        self,
        topic: str,
        level: str = None,
        branch: str = None,
        n: int = 10,
    ) -> list[dict]:
        """
        Return top-n courses most relevant to the topic query.

        Parameters
        ----------
        topic  : str   Query string. E.g. "machine learning", "python basics".
        level  : str   Optional. One of "Beginner", "Intermediate", "Advanced".
                       If no results found at the specified level, falls back to all.
        branch : str   Optional. If "CSE", applies 1.2× score boost to CS/ML courses.
        n      : int   Number of results to return (1–20).

        Returns
        -------
        list of dicts:
          course_title, subject, level, rating, course_url, score
        """
        # Vectorise the query using the fitted TF-IDF vocabulary
        query_vec = self.vectorizer.transform([topic.lower().strip()])

        # Compute cosine similarity between query and all courses
        scores = cosine_similarity(query_vec, self.tf_matrix).flatten()

        result_df = self.df.copy()
        result_df["score"] = scores

        # Apply level filter
        if level:
            filtered = result_df[
                result_df["level"].str.lower() == level.lower()
            ].copy()
            # Fall back to all levels if filter returns empty results
            result_df = filtered if len(filtered) > 0 else result_df

        # Apply CSE branch boost
        if branch and branch.upper() == "CSE":
            cs_mask = result_df["subject"].str.contains(
                self._CS_PATTERN, case=False, na=False
            )
            result_df.loc[cs_mask, "score"] *= 1.2

        # Sort and take top-n
        top = result_df.nlargest(n, "score")

        return [
            {
                "course_title": row["course_title"],
                "subject"     : row["subject"],
                "level"       : row["level"],
                "rating"      : round(float(row["rating"]), 2),
                "course_url"  : row.get("course_url", ""),
                "score"       : round(float(row["score"]), 4),
            }
            for _, row in top.iterrows()
        ]

    def get_popular(self, n: int = 30) -> list[dict]:
        """
        Return top-n courses by rating.
        Used by GET /api/v1/courses/popular.

        Parameters
        ----------
        n : int  Number of courses to return (default 30).

        Returns
        -------
        list of dicts — course metadata sorted by rating descending.
        """
        top = self.df.nlargest(n, "rating")
        return [
            {
                "course_title": row["course_title"],
                "subject"     : row["subject"],
                "level"       : row["level"],
                "rating"      : round(float(row["rating"]), 2),
                "course_url"  : row.get("course_url", ""),
            }
            for _, row in top.iterrows()
        ]

    def autocomplete(self, q: str, limit: int = 8) -> list[str]:
        """
        Return course titles containing q as a substring.
        Used by the course search bar dropdown.

        Parameters
        ----------
        q     : str  Query string (minimum 2 characters).
        limit : int  Maximum suggestions (default 8).

        Returns
        -------
        list of matching course title strings.
        """
        if not q or len(q.strip()) < 2:
            return []
        q_lower = q.strip().lower()
        return [t for t in self.all_titles if q_lower in t.lower()][:limit]

    def get_levels(self) -> list[str]:
        """
        Return the unique difficulty levels present in the dataset.
        Used to dynamically populate filter buttons in the frontend.
        """
        return sorted(self.df["level"].dropna().unique().tolist())

    def get_subjects(self) -> list[str]:
        """
        Return the unique subjects present in the dataset.
        """
        return sorted(self.df["subject"].dropna().unique().tolist())
