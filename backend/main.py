import sys, traceback

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from contextlib import asynccontextmanager
    import uvicorn

    from core.engine import RecommendationEngine
    from routers import recommend, books, health

    engine = RecommendationEngine()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("🚀 Starting engine…", flush=True)
        try:
            await engine.initialize()
        except Exception as e:
            print(f"❌ Engine init failed: {e}", flush=True)
            traceback.print_exc()
            raise
        app.state.engine = engine
        yield
        print("🛑 Shutting down.", flush=True)

    app = FastAPI(
        title="Hybrid Book Recommendation API",
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

except Exception as e:
    print(f"❌ Startup error: {e}", flush=True)
    traceback.print_exc()
    sys.exit(3)

if __name__ == "__main__":
    import os
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
