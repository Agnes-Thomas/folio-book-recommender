# 📚 Folio — Hybrid Book Recommendation System

A production-ready deployment of the GoodBooks-10k hybrid recommendation system.
Converts the Google Colab notebook into a full-stack web application.

```
book-recommender/
├── backend/          # FastAPI Python API
│   ├── main.py
│   ├── core/
│   │   └── engine.py         ← all ML logic (FAISS, SVD, meta-reg)
│   ├── routers/
│   │   ├── recommend.py      ← POST /api/recommend
│   │   ├── books.py          ← GET  /api/search
│   │   └── health.py         ← GET  /health, /api/stats
│   ├── models/schemas.py
│   └── requirements.txt
├── frontend/         # React + Vite UI
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── SearchBar.jsx
│   │   │   ├── SeedCard.jsx
│   │   │   └── BookCard.jsx
│   │   └── hooks/useRecommendations.js
│   └── package.json
└── docker-compose.yml
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Content model | `all-MiniLM-L6-v2` + FAISS IndexFlatIP |
| Collaborative filter | SVD (scikit-surprise), trained on 500k real ratings |
| Calibrator | LinearRegression meta-model |
| Cold-start | Open Library Search API |
| Backend | FastAPI + Uvicorn |
| Frontend | React 18 + Vite |
| Styling | Pure CSS with CSS variables |

---

## Quick Start (Local)

### Option A — Docker Compose (recommended)

```bash
docker compose up --build
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

> First start downloads GoodBooks CSVs (~80 MB) and trains models (~3–5 min).

---

### Option B — Manual

**Backend**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # edit if needed
python main.py
```

Backend runs on http://localhost:8000

**Frontend** (new terminal)

```bash
cd frontend
npm install
cp .env.example .env             # leave VITE_API_URL empty for dev proxy
npm run dev
```

Frontend runs on http://localhost:5173

---

## API Reference

### `POST /api/recommend`

```json
{
  "query": "The Hunger Games",
  "user_id": 42,
  "top_n": 5
}
```

**Response**
```json
{
  "seed": { "title": "...", "authors": "...", "cover_url": "...", "source": "LOCAL" },
  "recommendations": [
    {
      "title": "Divergent",
      "authors": "Veronica Roth",
      "final_score": 0.8431,
      "content_score": 0.7820,
      "collab_score": 0.7200,
      "explanation": "CF-heavy blend (user_weight=0.84, pop=0.76)",
      "source": "LOCAL"
    }
  ]
}
```

### `GET /api/search?q=hunger&limit=6`

Returns GoodBooks catalogue matches for autocomplete.

### `GET /health`

Returns `{ "status": "ok", "ready": true/false }` — use to poll engine warm-up.

### `GET /api/stats`

Returns model metadata (book count, rating count, embedding dim, etc.)

---

## Deployment

### Render.com (free tier)

1. Push this repo to GitHub.
2. Create a **Web Service** pointing to `backend/`, runtime Python 3.11.
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Create a **Static Site** pointing to `frontend/`.
   - Build command: `npm install && npm run build`
   - Publish directory: `dist`
   - Set env var `VITE_API_URL=https://your-backend.onrender.com`

### Railway / Fly.io

Same pattern — backend as a Python service, frontend as a static build.
Use the included `docker-compose.yml` for single-machine deployments.

---

## Environment Variables

**Backend (`.env`)**

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `goodbooks_data` | Where CSVs are stored |
| `SVD_SAMPLE` | `500000` | Rows used for SVD training (0 = all 6M) |
| `PORT` | `8000` | Server port |

**Frontend (`.env`)**

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_URL` | `` (empty) | Backend URL (empty = use Vite proxy) |

---

## Notes

- **First startup is slow** (~3–5 min): downloads data, encodes 10k embeddings, trains SVD.
  Subsequent starts reuse downloaded CSVs (only re-trains models).
- User IDs 1–53424 correspond to real GoodBooks users.
  New/unknown IDs are treated as cold-start (content-only recommendations).
- The Open Library fallback handles any book title not in the GoodBooks catalogue.
