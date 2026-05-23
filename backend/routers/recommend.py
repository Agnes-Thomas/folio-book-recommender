from fastapi import APIRouter, Request, HTTPException
from models.schemas import RecommendationRequest, RecommendationResponse, SeedBook, BookResult

router = APIRouter()

@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: Request, body: RecommendationRequest):
    engine = request.app.state.engine
    if not engine.ready:
        raise HTTPException(status_code=503, detail="Engine still loading, please retry in a moment.")
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not (1 <= body.top_n <= 15):
        raise HTTPException(status_code=400, detail="top_n must be between 1 and 15.")

    seed, recs = engine.smart_recommend(body.query, user_id=body.user_id, top_n=body.top_n)
    return RecommendationResponse(
        seed=SeedBook(**seed) if seed else None,
        recommendations=[BookResult(**r) for r in recs],
    )
