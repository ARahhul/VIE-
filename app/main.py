from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.ingest import router as ingest_router
from app.api.jobs import router as jobs_router
from app.db.base import init_db
from app.jobs.queue import start_worker, stop_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    start_worker()
    yield
    stop_worker()


app = FastAPI(title="VigilNetra Investigation Engine", version="0.1.0", lifespan=lifespan)
app.include_router(ingest_router)
app.include_router(jobs_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
