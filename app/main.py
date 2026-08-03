from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.device_profiles import router as device_profiles_router
from app.api.incidents import router as incidents_router
from app.api.ingest import router as ingest_router
from app.api.jobs import router as jobs_router
from app.core.tracing import configure_tracing
from app.db.base import init_db
from app.jobs.queue import start_worker, stop_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_tracing()
    init_db()
    start_worker()
    yield
    stop_worker()


app = FastAPI(title="VigilNetra Investigation Engine", version="0.1.0", lifespan=lifespan)
app.include_router(ingest_router)
app.include_router(jobs_router)
app.include_router(device_profiles_router)
app.include_router(incidents_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
