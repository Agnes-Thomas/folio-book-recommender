from fastapi import APIRouter, Request, Query
from models.schemas import SearchResult
from typing import List

router = APIRouter()

@router.get("/search", response_model=List[SearchResult])
async def search_books(request: Request, q: str = Query(..., min_length=2), limit: int = 8):
    engine = request.app.state.engine
    if not engine.ready:
        return []
    results = engine.search_titles(q, limit=limit)
    return [SearchResult(**r) for r in results]
