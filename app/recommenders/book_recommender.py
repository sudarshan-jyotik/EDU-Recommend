"""
app/recommenders/book_recommender.py  — VERSION 2
Improvements: rating-weighted re-ranking, larger candidate pool (n+21),
per-user normalised ratings, partial-word title matching, and portable
version-independent loading (.npz + .json.gz or .pkl fallback).
"""

import gzip
import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from sklearn.neighbors import NearestNeighbors


class BookRecommender:
    """Item-based KNN collaborative filter for book recommendations."""

    def __init__(self, model_dir: str = "models"):
        bundle_npz = os.path.join(model_dir, "books_bundle.npz")
        bundle_meta = os.path.join(model_dir, "books_meta.json.gz")

        self._avg_rating_map: dict[str, float] = {}
        self._meta_map: dict[str, dict[str, str]] = {}
        self._popular_records: list[dict] = []

        if os.path.exists(bundle_npz) and os.path.exists(bundle_meta):
            self._init_from_bundle(bundle_npz, bundle_meta)
        else:
            self._init_from_pickles(model_dir)

        self._global_mean = (
            float(np.mean(list(self._avg_rating_map.values())))
            if self._avg_rating_map
            else 5.0
        )
        self._title_to_idx = {t: i for i, t in enumerate(self.all_titles)}
        self._lower_map = {t.lower(): t for t in self.all_titles}
        print(f"[BookRecommender v2] {len(self.all_titles)} books  sparse={self.sparse.shape}")

    def _init_from_bundle(self, bundle_npz: str, bundle_meta: str) -> None:
        self.sparse = load_npz(bundle_npz).astype(np.float32)
        with gzip.open(bundle_meta, "rt", encoding="utf-8") as f:
            meta_obj = json.load(f)

        self.all_titles: list[str] = meta_obj["titles"]
        self._popular_records = meta_obj.get("popular", [])
        self._meta_map = meta_obj.get("meta", {})
        for row in self._popular_records:
            t = str(row.get("Book-Title", ""))
            if t:
                self._avg_rating_map[t] = float(row.get("avg_rating", row.get("score", 5.0)))

        self.model = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=min(31, max(1, len(self.all_titles))),
        )
        self.model.fit(self.sparse)

    def _init_from_pickles(self, model_dir: str) -> None:
        def _load(name: str):
            path = os.path.join(model_dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing model file: {path}")
            with open(path, "rb") as f:
                return pickle.load(f)

        pivot = _load("book_pivot.pkl")
        popular_df = _load("popular_df.pkl")
        meta = _load("books_meta.pkl")

        self.sparse = csr_matrix(pivot.values, dtype=np.float32)
        self.all_titles = pivot.index.tolist()

        try:
            self.model = _load("book_model.pkl")
        except Exception:
            self.model = NearestNeighbors(
                metric="cosine",
                algorithm="brute",
                n_neighbors=min(31, max(1, len(self.all_titles))),
            )
            self.model.fit(self.sparse)

        title_col = "Book-Title" if "Book-Title" in popular_df.columns else popular_df.columns[0]
        for _, row in popular_df.iterrows():
            t = str(row.get(title_col, row.iloc[0]))
            self._avg_rating_map[t] = float(row.get("avg_rating", row.get("score", 5.0)))

        self._popular_records = popular_df.to_dict("records")
        for row in self._popular_records:
            raw = row.get("Image-URL-M", row.get("cover", ""))
            row["cover"] = str(raw) if str(raw).startswith("http") else ""

        meta_idx = meta.set_index("Book-Title") if "Book-Title" in meta.columns else meta
        for title in self.all_titles:
            try:
                row = meta_idx.loc[title]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                author = self._safe_str(row.get("Book-Author", ""), "Unknown")
                cover = self._safe_str(row.get("Image-URL-M", ""), "")
                self._meta_map[title] = {
                    "author": author,
                    "cover": cover if cover.startswith("http") else "",
                }
            except KeyError:
                pass

    def recommend(self, book_name: str, n: int = 10) -> list[dict]:
        """KNN + rating-weighted re-ranking (70% cosine + 30% rating quality)."""
        matched = self._resolve_title(book_name)
        if matched is None:
            return []

        idx = self._title_to_idx[matched]
        vec = self.sparse[idx]
        pool = min(n + 21, len(self.all_titles))
        distances, indices = self.model.kneighbors(vec, n_neighbors=pool)

        candidates = []
        for dist, i in zip(distances[0], indices[0]):
            title = self.all_titles[int(i)]
            if title == matched:
                continue
            cos_sim = max(0.0, 1.0 - float(dist))
            raw_avg = self._avg_rating_map.get(title, self._global_mean)
            norm_r = min(1.0, raw_avg / 10.0) if raw_avg > 1.0 else raw_avg
            blended = 0.70 * cos_sim + 0.30 * norm_r
            author, cover = self._get_meta(title)
            candidates.append(
                {
                    "title": title,
                    "author": author,
                    "cover": cover,
                    "similarity": round(cos_sim * 100.0, 1),
                    "_blended": blended,
                }
            )

        candidates.sort(key=lambda x: x["_blended"], reverse=True)
        for c in candidates:
            c.pop("_blended", None)
        return candidates[:n]

    def get_popular(self, n: int = 50) -> list[dict]:
        return [dict(row) for row in self._popular_records[:n]]

    def autocomplete(self, q: str, limit: int = 10) -> list[str]:
        if not q or len(q.strip()) < 2:
            return []
        ql = q.strip().lower()
        prefix = [t for t in self.all_titles if t.lower().startswith(ql)]
        substr = [t for t in self.all_titles if ql in t.lower() and not t.lower().startswith(ql)]
        return (prefix + substr)[:limit]

    def _resolve_title(self, book_name: str):
        cleaned = (book_name or "").strip()
        if not cleaned:
            return None
        if cleaned in self._title_to_idx:
            return cleaned
        r = self._lower_map.get(cleaned.lower())
        if r:
            return r
        words = cleaned.lower().split()
        if len(words) >= 2:
            for title in self.all_titles:
                tl = title.lower()
                if all(w in tl for w in words):
                    return title
        return None

    def _get_meta(self, title: str) -> tuple[str, str]:
        info = self._meta_map.get(title)
        if not info:
            return "Unknown", ""
        return info.get("author", "Unknown") or "Unknown", info.get("cover", "") or ""

    @staticmethod
    def _safe_str(val, fallback: str = "") -> str:
        if val is None:
            return fallback
        try:
            if pd.isna(val):
                return fallback
        except (TypeError, ValueError):
            pass
        s = str(val).strip()
        return s if s and s.lower() != "nan" else fallback
