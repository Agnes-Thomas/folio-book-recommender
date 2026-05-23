from pydantic import BaseModel
from typing import Optional, List


class RecommendationRequest(BaseModel):
    query: str
    user_id: int = 1
    top_n: int = 5


class BookResult(BaseModel):
    book_idx: int
    title: str
    authors: Optional[str]
    average_rating: Optional[float]
    isbn: Optional[str]
    cover_url: Optional[str]
    publication_year: Optional[int]
    content_score: Optional[float]
    collab_score: Optional[float]
    final_score: float
    explanation: str
    source: str


class SeedBook(BaseModel):
    title: str
    authors: Optional[str]
    cover_url: Optional[str]
    average_rating: Optional[float] = None
    source: str


class RecommendationResponse(BaseModel):
    seed: Optional[SeedBook]
    recommendations: List[BookResult]


class SearchResult(BaseModel):
    title: str
    authors: Optional[str]
    book_idx: int


class ModelStats(BaseModel):
    books: int
    ratings: int
    embedding_dim: int
    svd_factors: int
    svd_epochs: int
    faiss_type: str
