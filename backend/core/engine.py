"""
Hybrid Book Recommendation Engine — Ultra-lightweight for Render Free (512MB)

Memory optimisations:
- Ratings CSV read in chunks, only user_activity dict kept (not full dataframe)
- TF-IDF limited to 15k features
- SVD sample capped at 50k
- scipy sparse throughout
- No sentence-transformers, no PyTorch, no FAISS
"""

import os, time, gc, asyncio
import requests
import numpy as np
import pandas as pd
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_split

DATA_DIR   = os.getenv("DATA_DIR", "goodbooks_data")
SVD_SAMPLE = int(os.getenv("SVD_SAMPLE", 50_000))

OL_COV_BASE = "https://covers.openlibrary.org"
OL_BASE     = "https://openlibrary.org"
OL_HEADERS  = {"User-Agent": "HybridBookRecommender/1.0"}
_CACHE: dict = {}


class RecommendationEngine:
    def __init__(self):
        self.ready          = False
        self.books: Optional[pd.DataFrame] = None
        self.tfidf_matrix   = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd_model      = None
        self.reg_model: Optional[LinearRegression] = None
        self.user_activity: dict = {}
        # small ratings sample kept only for meta-regression
        self._ratings_sample: Optional[pd.DataFrame] = None

    async def initialize(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_all)
        self.ready = True
        print("✅ Engine ready.")

    # ------------------------------------------------------------------ #

    def _load_all(self):
        self._download_data()
        self._load_books()
        self._build_tfidf()
        self._load_ratings_light()
        self._train_svd()
        self._train_meta()
        gc.collect()

    # ------------------------------------------------------------------ #
    #  Data download                                                       #
    # ------------------------------------------------------------------ #

    def _download_data(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        urls = {
            "books.csv":   "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv",
            "ratings.csv": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv",
        }
        for fname, url in urls.items():
            dest = os.path.join(DATA_DIR, fname)
            if os.path.exists(dest):
                continue
            print(f"  Downloading {fname}…")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            open(dest, "wb").write(r.content)

    # ------------------------------------------------------------------ #
    #  Books (small — always fits)                                        #
    # ------------------------------------------------------------------ #

    def _load_books(self):
        keep = ["book_id", "title", "authors", "average_rating",
                "ratings_count", "isbn", "original_publication_year", "image_url"]
        raw = pd.read_csv(os.path.join(DATA_DIR, "books.csv"), usecols=keep)
        self.books = raw.dropna(subset=["title"]).reset_index(drop=True)

        scaler = MinMaxScaler()
        self.books["popularity_score"] = scaler.fit_transform(
            self.books["ratings_count"].fillna(0).values.reshape(-1, 1)
        ).astype("float32")

        self.books["text_feature"] = (
            self.books["title"].fillna("") + " " +
            self.books["authors"].fillna("") + " " +
            self.books["authors"].fillna("")
        )
        # keep only columns we actually use later
        self.books = self.books[["book_id","title","authors","average_rating",
                                  "ratings_count","isbn","original_publication_year",
                                  "image_url","popularity_score","text_feature"]]
        print(f"  Books loaded: {len(self.books):,}")

    # ------------------------------------------------------------------ #
    #  TF-IDF content model                                               #
    # ------------------------------------------------------------------ #

    def _build_tfidf(self):
        print("  Building TF-IDF…")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=15_000,   # smaller → less RAM
            sublinear_tf=True,
            dtype=np.float32,
        )
        texts = self.books["text_feature"].fillna("").tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"  ✅ TF-IDF {self.tfidf_matrix.shape}")

    def _encode_query(self, text: str):
        return self.vectorizer.transform([text])

    def _content_recommend(self, query_vec, top_n=10, exclude_idx=None):
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        if exclude_idx is not None:
            sims[exclude_idx] = -1
        top_idx = np.argsort(sims)[::-1][:top_n + 5]
        results = []
        for i in top_idx:
            if i == exclude_idx:
                continue
            results.append((int(i), float(sims[i])))
            if len(results) >= top_n:
                break
        return results

    # ------------------------------------------------------------------ #
    #  Ratings — chunked, keep only what we need                         #
    # ------------------------------------------------------------------ #

    def _load_ratings_light(self):
        print("  Reading ratings (chunked)…")
        path = os.path.join(DATA_DIR, "ratings.csv")

        # 1. Build user_activity from full file in chunks (no full DF in memory)
        activity: dict = {}
        for chunk in pd.read_csv(path, usecols=["user_id"], chunksize=200_000):
            for uid, cnt in chunk["user_id"].value_counts().items():
                activity[uid] = activity.get(uid, 0) + cnt
        self.user_activity = activity
        gc.collect()

        # 2. Build book_id → book_idx map
        id2idx = {bid: idx for idx, bid in enumerate(self.books["book_id"])}

        # 3. Sample only SVD_SAMPLE rows for training
        print(f"  Sampling {SVD_SAMPLE:,} ratings for SVD…")
        chunks = []
        remaining = SVD_SAMPLE
        for chunk in pd.read_csv(path, usecols=["user_id","book_id","rating"], chunksize=200_000):
            chunk = chunk[chunk["book_id"].isin(id2idx)]
            chunk["book_idx"] = chunk["book_id"].map(id2idx)
            take = min(remaining, len(chunk))
            chunks.append(chunk.sample(take, random_state=42))
            remaining -= take
            if remaining <= 0:
                break
        self._ratings_sample = pd.concat(chunks, ignore_index=True)
        print(f"  ✅ Ratings sample: {len(self._ratings_sample):,}")

    # ------------------------------------------------------------------ #
    #  SVD                                                                #
    # ------------------------------------------------------------------ #

    def _train_svd(self):
        print("  Training SVD…")
        reader = Reader(rating_scale=(1, 5))
        data   = Dataset.load_from_df(
            self._ratings_sample[["user_id","book_idx","rating"]], reader
        )
        self.trainset, self.testset = surprise_split(data, test_size=0.2, random_state=42)
        self.svd_model = SVD(n_factors=50, n_epochs=15, lr_all=0.005, reg_all=0.02)
        self.svd_model.fit(self.trainset)
        print("  ✅ SVD trained.")

    # ------------------------------------------------------------------ #
    #  Meta-regression                                                    #
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
        test_preds  = self.svd_model.test(self.testset)
        Xtr, ytr = self._make_meta_features(train_preds)
        Xte, yte = self._make_meta_features(test_preds)
        self.reg_model = LinearRegression().fit(Xtr, ytr)
        r2 = self.reg_model.score(Xte, yte)
        # free the sample now that training is done
        self._ratings_sample = None
        gc.collect()
        print(f"  ✅ Meta-regression R²={r2:.4f}")

    # ------------------------------------------------------------------ #
    #  Scoring                                                            #
    # ------------------------------------------------------------------ #

    def _compute_user_weight(self, user_id: int) -> float:
        return min(self.user_activity.get(user_id, 0) / 50, 1.0)

    def _score_candidates(self, candidates, user_id: int):
        uw  = self._compute_user_weight(user_id)
        act = self.user_activity.get(user_id, 0)
        results = []
        for idx, cos_sim in candidates:
            row        = self.books.iloc[idx]
            pop        = float(row["popularity_score"])
            collab_est = self.svd_model.predict(user_id, idx).est
            hybrid     = uw * (collab_est / 5) + (1 - uw) * cos_sim
            context    = hybrid + 0.1 * pop
            reg_pred   = float(self.reg_model.predict([[collab_est/5, pop, min(act,500)/500]])[0])
            final      = round(0.7 * context + 0.3 * (reg_pred / 5), 4)
            results.append({
                "book_idx":         idx,
                "title":            row["title"],
                "authors":          row["authors"],
                "average_rating":   float(row["average_rating"]) if pd.notna(row["average_rating"]) else None,
                "isbn":             str(row["isbn"]) if pd.notna(row["isbn"]) else None,
                "cover_url":        self._cover_url(row),
                "publication_year": int(row["original_publication_year"])
                                    if pd.notna(row["original_publication_year"]) else None,
                "content_score":    round(float(cos_sim), 4),
                "collab_score":     round(collab_est / 5, 4),
                "final_score":      final,
                "explanation":      f"{'CF-heavy' if uw>0.6 else 'Content-heavy'} "
                                    f"(user_weight={uw:.2f}, pop={pop:.2f})",
                "source":           "LOCAL",
            })
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results

    # ------------------------------------------------------------------ #
    #  Open Library cold-start                                            #
    # ------------------------------------------------------------------ #

    def _ol_get(self, url, params=None):
        key = url + str(sorted((params or {}).items()))
        if key in _CACHE:
            return _CACHE[key]
        time.sleep(0.35)
        try:
            r    = requests.get(url, params=params, headers=OL_HEADERS, timeout=10)
            data = r.json() if r.ok else None
        except Exception:
            data = None
        _CACHE[key] = data
        return data

    def _cover_url(self, row, size="M"):
        isbn = str(row.get("isbn","") or "")
        if isbn and isbn not in ("nan",""):
            return f"{OL_COV_BASE}/b/isbn/{isbn}-{size}.jpg"
        img = str(row.get("image_url","") or "")
        return img if img not in ("nan","") else None

    def _cold_start_search(self, query: str, top_n: int = 5):
        data = self._ol_get(f"{OL_BASE}/search.json", params={"q": query, "limit": 1})
        if not data or not data.get("docs"):
            return None, []
        doc   = data["docs"][0]
        title = doc.get("title", query)
        authors = ", ".join(doc.get("author_name", []))
        isbn  = (doc.get("isbn") or [None])[0]
        seed  = {"title": title, "authors": authors,
                 "cover_url": f"{OL_COV_BASE}/b/isbn/{isbn}-M.jpg" if isbn else None,
                 "source": "OPEN LIBRARY"}
        qv   = self._encode_query(f"{title} {authors}")
        cands = self._content_recommend(qv, top_n=top_n)
        recs  = []
        for idx, cos_sim in cands:
            row = self.books.iloc[idx]
            recs.append({
                "book_idx": idx, "title": row["title"], "authors": row["authors"],
                "average_rating": float(row["average_rating"]) if pd.notna(row["average_rating"]) else None,
                "isbn": str(row["isbn"]) if pd.notna(row["isbn"]) else None,
                "cover_url": self._cover_url(row),
                "publication_year": int(row["original_publication_year"])
                                    if pd.notna(row["original_publication_year"]) else None,
                "content_score": round(float(cos_sim), 4), "collab_score": None,
                "final_score": round(float(cos_sim), 4),
                "explanation": "Content-only (cold-start)", "source": "LOCAL",
            })
        return seed, recs

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def search_titles(self, query: str, limit: int = 8):
        mask = self.books["title"].str.lower().str.contains(query.lower(), na=False)
        return [{"title": r["title"], "authors": r["authors"], "book_idx": int(i)}
                for i, r in self.books[mask].head(limit).iterrows()]

    def smart_recommend(self, query: str, user_id: int = 1, top_n: int = 5):
        mask       = self.books["title"].str.lower().str.contains(query.lower(), na=False)
        local_hits = self.books[mask]
        if local_hits.empty:
            return self._cold_start_search(query, top_n)
        seed_row = local_hits.iloc[0]
        seed_idx = int(local_hits.index[0])
        seed = {"title": seed_row["title"], "authors": seed_row["authors"],
                "cover_url": self._cover_url(seed_row),
                "average_rating": float(seed_row["average_rating"])
                                  if pd.notna(seed_row["average_rating"]) else None,
                "source": "LOCAL"}
        qv    = self._encode_query(seed_row["text_feature"])
        cands = self._content_recommend(qv, top_n=top_n+5, exclude_idx=seed_idx)
        return seed, self._score_candidates(cands, user_id)[:top_n]

    def get_model_stats(self):
        return {"books": len(self.books), "ratings_sample": SVD_SAMPLE,
                "tfidf_features": self.tfidf_matrix.shape[1],
                "svd_factors": 50, "faiss_type": "TF-IDF cosine (lightweight)"}
