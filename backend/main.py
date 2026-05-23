"""
Hybrid Book Recommendation System — FastAPI Backend
Converts the GoodBooks-10k Colab notebook into a deployable REST API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from core.engine import RecommendationEngine
from routers import recommend, books, health

engine = RecommendationEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup."""
    print("🚀 Starting Hybrid Book Recommendation Engine…")
    await engine.initialize()
    app.state.engine = engine
    yield
    print("🛑 Shutting down.")


app = FastAPI(
    title="Hybrid Book Recommendation API",
    description="Content-Based + SVD Collaborative Filtering + Meta-Regression on GoodBooks-10k",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(recommend.router, prefix="/api", tags=["Recommendations"])
app.include_router(books.router, prefix="/api", tags=["Books"])


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
