"""
app/recommenders/book_recommender.py
=====================================
BookRecommender class — wraps the 4 book ML pickles.

Created on Day 5 of the EduRecommend BTP project.
Extracted from notebooks/02_books_pipeline.ipynb.

This class is the only place in the codebase that touches
the book ML pickles. FastAPI routes call methods on this class
via dependency injection (see app/dependencies.py).

Loaded at startup by load_models() in app/dependencies.py.
"""

import pickle
import os
import pandas as pd
from scipy.sparse import csr_matrix


class BookRecommender:
    """
    Item-based collaborative filter for book recommendations.

    Uses KNN with cosine similarity fitted on a user-book rating matrix.
    Supports:
      - recommend()    — top-n similar books for a given title
      - get_popular()  — top-n books by Bayesian popularity score
      - autocomplete() — matching titles for the search dropdown
    """

    def __init__(self, model_dir: str = "models"):
        """
        Load all 4 book pickles from model_dir.
        Called once at app startup via dependencies.load_models().

        Parameters
        ----------
        model_dir : str
            Directory containing the 4 pickle files.
            Default matches the Docker WORKDIR layout.
        """
        def _load(filename):
            path = os.path.join(model_dir, filename)
            with open(path, "rb") as f:
                return pickle.load(f)

        self.model      = _load("book_model.pkl")    # NearestNeighbors(cosine)
        self.pivot      = _load("book_pivot.pkl")    # DataFrame: books × users
        self.popular_df = _load("popular_df.pkl")    # Bayesian popularity table
        self.meta       = _load("books_meta.pkl")    # Title → Author + cover URL

        # Pre-build sparse matrix — converting from pivot once is faster
        # than doing it on every request
        self.sparse = csr_matrix(self.pivot)

        # Title list for autocomplete — kept as a Python list for fast iteration
        self.all_titles = self.pivot.index.tolist()

        # Index books_meta on Book-Title for O(1) cover/author lookup
        # .set_index() does NOT modify the original pickle in memory
        if "Book-Title" in self.meta.columns:
            self.meta_idx = self.meta.set_index("Book-Title")
        else:
            self.meta_idx = self.meta  # already indexed

        print(
            f"[BookRecommender] loaded — "
            f"{len(self.all_titles)} books, "
            f"pivot={self.pivot.shape}, "
            f"has_covers={'Image-URL-M' in self.meta.columns}"
        )

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def recommend(self, book_name: str, n: int = 10) -> list[dict]:
        """
        Return top-n book recommendations similar to book_name.

        Parameters
        ----------
        book_name : str  Exact book title as stored in the pivot index.
        n         : int  Number of results to return (1–20).

        Returns
        -------
        list of dicts with keys: title, author, cover, similarity
        Returns [] if book_name is not in the model.
        """
        # Normalise — try exact match first, then case-insensitive
        matched_name = self._resolve_title(book_name)
        if matched_name is None:
            return []

        idx = self.pivot.index.get_loc(matched_name)
        vec = self.sparse[idx]

        # n+1 because index 0 is always the query book itself
        distances, indices = self.model.kneighbors(
            vec, n_neighbors=min(n + 1, len(self.all_titles))
        )

        results = []
        for dist, i in zip(distances[0][1:], indices[0][1:]):
            title         = self.pivot.index[i]
            author, cover = self._get_meta(title)
            results.append({
                "title"     : title,
                "author"    : author,
                "cover"     : cover,
                "similarity": round((1.0 - float(dist)) * 100.0, 1),
            })

        return results

    def get_popular(self, n: int = 50) -> list[dict]:
        """
        Return top-n books by Bayesian popularity score.
        Used by the homepage (GET /api/v1/books/popular).

        Parameters
        ----------
        n : int  Number of books to return (default 50).

        Returns
        -------
        list of dicts — each row from popular_df converted to dict.
        Cover URL is normalised (empty string if missing or invalid).
        """
        rows = self.popular_df.head(n).to_dict("records")
        for row in rows:
            raw = row.get("Image-URL-M", "")
            row["cover"] = self._safe_str(raw) if str(raw).startswith("http") else ""
        return rows

    def autocomplete(self, q: str, limit: int = 10) -> list[str]:
        """
        Return book titles containing q as a substring (case-insensitive).
        Used by GET /api/v1/books/autocomplete?q=...

        Parameters
        ----------
        q     : str  Query string (minimum 2 characters).
        limit : int  Maximum number of suggestions (default 10).

        Returns
        -------
        list of matching title strings.
        """
        if not q or len(q.strip()) < 2:
            return []
        q_lower = q.strip().lower()
        return [t for t in self.all_titles if q_lower in t.lower()][:limit]

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    def _resolve_title(self, book_name: str):
        """
        Return the canonical title stored in the pivot index.
        Tries exact match first, then case-insensitive match.
        Returns None if no match found.
        """
        if book_name in self.pivot.index:
            return book_name

        lower_map = {t.lower(): t for t in self.all_titles}
        return lower_map.get(book_name.lower())

    def _get_meta(self, title: str) -> tuple[str, str]:
        """
        Look up Book-Author and Image-URL-M for a given title.
        Returns ("Unknown", "") if the title is not in books_meta.

        Uses .loc[] on the pre-indexed meta_idx DataFrame — O(1) lookup
        vs O(n) scan with boolean masking.
        """
        try:
            row = self.meta_idx.loc[title]
            # .loc may return a DataFrame if there are duplicate titles
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            author = self._safe_str(row.get("Book-Author", ""), "Unknown")
            cover  = self._safe_str(row.get("Image-URL-M", ""), "")
            cover  = cover if cover.startswith("http") else ""
            return author, cover
        except KeyError:
            return "Unknown", ""

    @staticmethod
    def _safe_str(val, fallback: str = "") -> str:
        """
        Safely convert a pandas value to string.
        Handles NaN, None, and the string "nan" that pandas sometimes produces.
        """
        if val is None:
            return fallback
        try:
            if pd.isna(val):
                return fallback
        except (TypeError, ValueError):
            pass
        s = str(val).strip()
        return s if s and s.lower() != "nan" else fallback
