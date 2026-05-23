"""
One-time pre-processing script — run this LOCALLY before deploying.

What it does:
  1. Downloads books.csv and ratings.csv from GoodBooks-10k
  2. Processes them into two tiny files:
     - backend/data/books_clean.csv     (~1.2 MB — 10k books)
     - backend/data/ratings_sample.csv  (~3 MB — 50k ratings)

These get committed to GitHub so Render never needs to
download or process the full 250MB ratings file.

Usage:
  pip install pandas requests tqdm
  python preprocess.py
"""

import os, requests, pandas as pd
from tqdm import tqdm

OUT_DIR = os.path.join("backend", "data")
os.makedirs(OUT_DIR, exist_ok=True)

BOOKS_URL   = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"
RATINGS_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv"
SAMPLE_N    = 50_000

def download(url, dest):
    if os.path.exists(dest):
        print(f"  {dest} already exists, skipping download.")
        return
    print(f"  Downloading {url} ...")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  ✅ Saved {dest}")

# ── Download ──────────────────────────────────────────────────────────
raw_books   = os.path.join(OUT_DIR, "_books_raw.csv")
raw_ratings = os.path.join(OUT_DIR, "_ratings_raw.csv")
download(BOOKS_URL,   raw_books)
download(RATINGS_URL, raw_ratings)

# ── Process books ────────────────────────────────────────────────────
print("\n  Processing books...")
books = pd.read_csv(raw_books, usecols=[
    "book_id","title","authors","average_rating",
    "ratings_count","isbn","original_publication_year","image_url"
]).dropna(subset=["title"]).reset_index(drop=True)

books["popularity_score"] = (
    books["ratings_count"].fillna(0) /
    books["ratings_count"].fillna(0).max()
).astype("float32")

books["text_feature"] = (
    books["title"].fillna("") + " " +
    books["authors"].fillna("") + " " +
    books["authors"].fillna("")
)

out_books = os.path.join(OUT_DIR, "books_clean.csv")
books.to_csv(out_books, index=False)
print(f"  ✅ Books: {len(books):,} rows → {out_books} ({os.path.getsize(out_books)//1024} KB)")

# ── Process ratings ──────────────────────────────────────────────────
print("\n  Processing ratings (building user_activity)...")
id2idx = {bid: idx for idx, bid in enumerate(books["book_id"])}

activity = {}
chunks_for_sample = []
remaining = SAMPLE_N

for chunk in pd.read_csv(raw_ratings, chunksize=200_000):
    # accumulate user activity
    for uid, cnt in chunk["user_id"].value_counts().items():
        activity[uid] = activity.get(uid, 0) + cnt
    # collect sample rows
    if remaining > 0:
        valid = chunk[chunk["book_id"].isin(id2idx)].copy()
        valid["book_idx"] = valid["book_id"].map(id2idx)
        take = min(remaining, len(valid))
        chunks_for_sample.append(valid.sample(take, random_state=42))
        remaining -= take

# Save ratings sample
sample = pd.concat(chunks_for_sample, ignore_index=True)[["user_id","book_idx","rating"]]
out_ratings = os.path.join(OUT_DIR, "ratings_sample.csv")
sample.to_csv(out_ratings, index=False)
print(f"  ✅ Ratings sample: {len(sample):,} rows → {out_ratings} ({os.path.getsize(out_ratings)//1024} KB)")

# Save user activity
import json
out_activity = os.path.join(OUT_DIR, "user_activity.json")
with open(out_activity, "w") as f:
    json.dump(activity, f)
print(f"  ✅ User activity: {len(activity):,} users → {out_activity} ({os.path.getsize(out_activity)//1024} KB)")

# Clean up raw files
os.remove(raw_books)
os.remove(raw_ratings)

print("\n✅ Pre-processing complete!")
print(f"   Now run: git add backend/data && git commit -m 'Add preprocessed data' && git push")
