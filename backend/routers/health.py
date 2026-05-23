from fastapi import APIRouter, Request

router = APIRouter()

@router.get("/health")
async def health(request: Request):
    engine = request.app.state.engine
    return {"status": "ok", "ready": engine.ready}

@router.get("/api/stats")
async def stats(request: Request):
    engine = request.app.state.engine
    if not engine.ready:
        return {"error": "Engine not ready yet"}
    return engine.get_model_stats()
