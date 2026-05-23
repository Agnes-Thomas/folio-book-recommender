"""
Core Recommendation Engine
Directly ported from the GoodBooks-10k Colab notebook.
All logic preserved: FAISS content model, SVD collaborative filter,
meta-regression calibrator, adaptive user weighting, cold-start fallback.
"""

import os
import time
import asyncio
import requests
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional

from sentence_transformers import SentenceTransformer
import faiss
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

DATA_DIR = os.getenv("DATA_DIR", "goodbooks_data")
SAMPLE = int(os.getenv("SVD_SAMPLE", 500_000))

OL_COV_BASE = "https://covers.openlibrary.org"
OL_BASE = "https://openlibrary.org"
OL_HEADERS = {"User-Agent": "HybridBookRecommender/1.0 (deploy@example.com)"}
_CACHE: dict = {}


class RecommendationEngine:
    def __init__(self):
        self.ready = False
        self.books: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None
        self.embed_model = None
        self.faiss_index = None
        self.svd_model = None
        self.reg_model = None
        self.user_activity: dict = {}
        self.dim: int = 0

    # ------------------------------------------------------------------ #
    #  Initialization                                                      #
    # ------------------------------------------------------------------ #

    async def initialize(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_all)
        self.ready = True
        print("✅ Recommendation engine ready.")

    def _load_all(self):
        self._download_data()
        self._load_data()
        self._build_embeddings()
        self._train_svd()
        self._train_meta()

    # ------------------------------------------------------------------ #
    #  Data Download & Load                                                #
    # ------------------------------------------------------------------ #

    def _download_data(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        files = {
            "books.csv": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv",
            "ratings.csv": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv",
        }
        for fname, url in files.items():
            dest = os.path.join(DATA_DIR, fname)
            if os.path.exists(dest):
                print(f"  {fname} already downloaded — skipping.")
                continue
            print(f"  Downloading {fname}…")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
            print(f"  ✅ {fname} saved.")

    def _load_data(self):
        books_raw = pd.read_csv(os.path.join(DATA_DIR, "books.csv"))
        keep = ["book_id", "title", "authors", "average_rating", "ratings_count",
                "isbn", "isbn13", "original_publication_year", "image_url"]
        self.books = books_raw[keep].copy().dropna(subset=["title"]).reset_index(drop=True)

        scaler = MinMaxScaler()
        self.books["popularity_score"] = scaler.fit_transform(
            self.books["ratings_count"].fillna(0).values.reshape(-1, 1)
        )
        self.books["text_feature"] = (
            self.books["title"].fillna("") + " " + self.books["authors"].fillna("")
        )

        ratings_raw = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
        self.ratings_df = ratings_raw.merge(
            self.books[["book_id"]].reset_index().rename(columns={"index": "book_idx"}),
            on="book_id",
        )
        self.user_activity = self.ratings_df["user_id"].value_counts().to_dict()
        print(f"  Books: {len(self.books):,}  |  Ratings: {len(self.ratings_df):,}")

    # ------------------------------------------------------------------ #
    #  Content Model (Sentence Embeddings + FAISS)                        #
    # ------------------------------------------------------------------ #

    def _build_embeddings(self):
        print("  Encoding embeddings (all-MiniLM-L6-v2)…")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = self.books["text_feature"].fillna("").tolist()
        embeddings = self.embed_model.encode(
            texts, show_progress_bar=True, batch_size=128
        ).astype("float32")
        faiss.normalize_L2(embeddings)
        self.dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(self.dim)
        self.faiss_index.add(embeddings)
        print(f"  ✅ FAISS index: {self.faiss_index.ntotal:,} vectors  |  dim={self.dim}")

    def _encode_query(self, text: str) -> np.ndarray:
        vec = self.embed_model.encode([text]).astype("float32")
        faiss.normalize_L2(vec)
        return vec

    def _content_recommend_from_vec(self, query_vec, top_n=5, exclude_idx=None):
        k = top_n + (1 if exclude_idx is not None else 0) + 10
        scores, indices = self.faiss_index.search(query_vec, k)
        results = []
        for i, s in zip(indices[0], scores[0]):
            if i == exclude_idx:
                continue
            results.append((int(i), float(s)))
            if len(results) >= top_n:
                break
        return results

    # ------------------------------------------------------------------ #
    #  Collaborative Filtering (SVD)                                      #
    # ------------------------------------------------------------------ #

    def _train_svd(self):
        print(f"  Training SVD on {min(SAMPLE, len(self.ratings_df)):,} ratings…")
        sample = (
            self.ratings_df.sample(SAMPLE, random_state=42)
            if SAMPLE and len(self.ratings_df) > SAMPLE
            else self.ratings_df
        )
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(sample[["user_id", "book_idx", "rating"]], reader)
        self.trainset, self.testset = surprise_split(data, test_size=0.20, random_state=42)
        self.svd_model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
        self.svd_model.fit(self.trainset)
        print("  ✅ SVD trained.")

    # ------------------------------------------------------------------ #
    #  Meta-Regression Calibrator                                         #
    # ------------------------------------------------------------------ #

    def _make_meta_features(self, preds):
        X, y = [], []
        for uid, iid, true_r, est, _ in preds:
            idx = int(iid)
            pop = float(self.books.iloc[idx]["popularity_score"]) if 0 <= idx < len(self.books) else 0
            act = min(self.user_activity.get(uid, 0), 500) / 500
            X.append([est / 5, pop, act])
            y.append(true_r)
        return X, y

    def _train_meta(self):
        train_preds = self.svd_model.test(self.trainset.build_testset())
        test_preds = self.svd_model.test(self.testset)
        X_train, y_train = self._make_meta_features(train_preds)
        X_test, y_test = self._make_meta_features(test_preds)
        self.reg_model = LinearRegression().fit(X_train, y_train)
        r2 = self.reg_model.score(X_test, y_test)
        print(f"  ✅ Meta-regression R²={r2:.4f}")

    # ------------------------------------------------------------------ #
    #  Scoring helpers                                                     #
    # ------------------------------------------------------------------ #

    def _compute_user_weight(self, user_id: int, threshold: int = 50) -> float:
        return min(self.user_activity.get(user_id, 0) / threshold, 1.0)

    def _score_candidates(self, candidate_indices, candidate_sims, user_id: int):
        uw = self._compute_user_weight(user_id)
        act = self.user_activity.get(user_id, 0)
        results = []
        for idx, cos_sim in zip(candidate_indices, candidate_sims):
            row = self.books.iloc[idx]
            pop = float(row["popularity_score"])
            collab_est = self.svd_model.predict(user_id, idx).est

            hybrid_score = uw * (collab_est / 5) + (1 - uw) * cos_sim
            context_score = hybrid_score + 0.1 * pop

            meta_feat = [[collab_est / 5, pop, min(act, 500) / 500]]
            reg_pred = float(self.reg_model.predict(meta_feat)[0])
            final_score = round(0.7 * context_score + 0.3 * (reg_pred / 5), 4)

            results.append({
                "book_idx": idx,
                "title": row["title"],
                "authors": row["authors"],
                "average_rating": float(row["average_rating"]) if pd.notna(row["average_rating"]) else None,
                "isbn": str(row["isbn"]) if pd.notna(row["isbn"]) else None,
                "cover_url": self._cover_url(row),
                "publication_year": int(row["original_publication_year"])
                    if pd.notna(row["original_publication_year"]) else None,
                "content_score": round(float(cos_sim), 4),
                "collab_score": round(collab_est / 5, 4),
                "final_score": final_score,
                "explanation": f"{'CF-heavy' if uw > 0.6 else 'Content-heavy'} blend "
                               f"(user_weight={uw:.2f}, pop={pop:.2f})",
                "source": "LOCAL",
            })
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results

    # ------------------------------------------------------------------ #
    #  Open Library / Cold-Start                                          #
    # ------------------------------------------------------------------ #

    def _ol_get(self, url, params=None):
        key = url + str(sorted((params or {}).items()))
        if key in _CACHE:
            return _CACHE[key]
        time.sleep(0.35)
        try:
            r = requests.get(url, params=params, headers=OL_HEADERS, timeout=10)
            data = r.json() if r.ok else None
        except Exception:
            data = None
        _CACHE[key] = data
        return data

    def _cover_url(self, row, size="M"):
        isbn = str(row.get("isbn", "") or "")
        if isbn and isbn not in ("nan", ""):
            return f"{OL_COV_BASE}/b/isbn/{isbn}-{size}.jpg"
        img = str(row.get("image_url", "") or "")
        return img if img not in ("nan", "") else None

    def _cold_start_search(self, query: str, top_n: int = 5):
        data = self._ol_get(f"{OL_BASE}/search.json", params={"q": query, "limit": 1})
        if not data or not data.get("docs"):
            return None, []
        doc = data["docs"][0]
        title = doc.get("title", query)
        authors = ", ".join(doc.get("author_name", []))
        isbn_list = doc.get("isbn", [])
        isbn = isbn_list[0] if isbn_list else None
        cover = f"{OL_COV_BASE}/b/isbn/{isbn}-M.jpg" if isbn else None

        seed = {"title": title, "authors": authors, "cover_url": cover, "source": "OPEN LIBRARY"}
        query_vec = self._encode_query(f"{title} {authors}")
        candidates = self._content_recommend_from_vec(query_vec, top_n=top_n)
        recs = []
        for idx, cos_sim in candidates:
            row = self.books.iloc[idx]
            recs.append({
                "book_idx": idx,
                "title": row["title"],
                "authors": row["authors"],
                "average_rating": float(row["average_rating"]) if pd.notna(row["average_rating"]) else None,
                "isbn": str(row["isbn"]) if pd.notna(row["isbn"]) else None,
                "cover_url": self._cover_url(row),
                "publication_year": int(row["original_publication_year"])
                    if pd.notna(row["original_publication_year"]) else None,
                "content_score": round(float(cos_sim), 4),
                "collab_score": None,
                "final_score": round(float(cos_sim), 4),
                "explanation": "Content-only (cold-start fallback)",
                "source": "LOCAL",
            })
        return seed, recs

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def search_titles(self, query: str, limit: int = 8):
        """Fuzzy title search within GoodBooks catalogue."""
        q = query.lower()
        mask = self.books["title"].str.lower().str.contains(q, na=False)
        results = self.books[mask].head(limit)
        return [
            {"title": r["title"], "authors": r["authors"], "book_idx": int(i)}
            for i, r in results.iterrows()
        ]

    def smart_recommend(self, query: str, user_id: int = 1, top_n: int = 5):
        """Main entry: try local catalogue first, fall back to Open Library."""
        q = query.lower()
        mask = self.books["title"].str.lower().str.contains(q, na=False)
        local_hits = self.books[mask]

        if local_hits.empty:
            seed, recs = self._cold_start_search(query, top_n)
            return seed, recs

        seed_row = local_hits.iloc[0]
        seed_idx = int(local_hits.index[0])
        seed = {
            "title": seed_row["title"],
            "authors": seed_row["authors"],
            "cover_url": self._cover_url(seed_row),
            "average_rating": float(seed_row["average_rating"]) if pd.notna(seed_row["average_rating"]) else None,
            "source": "LOCAL",
        }

        query_vec = self._encode_query(seed_row["text_feature"])
        candidates = self._content_recommend_from_vec(query_vec, top_n=top_n + 5, exclude_idx=seed_idx)
        idxs = [c[0] for c in candidates]
        sims = [c[1] for c in candidates]
        recs = self._score_candidates(idxs, sims, user_id)[:top_n]
        return seed, recs

    def get_model_stats(self):
        return {
            "books": len(self.books),
            "ratings": len(self.ratings_df),
            "embedding_dim": self.dim,
            "svd_factors": 100,
            "svd_epochs": 20,
            "faiss_type": "IndexFlatIP",
        }
