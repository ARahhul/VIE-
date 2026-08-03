import asyncio
import logging

from app.db.base import SessionLocal
from app.db.models import ReportJob
from app.graph.build import compiled_graph
from app.graph.state import InvestigationState

logger = logging.getLogger("vie.jobs")

_queue: asyncio.Queue[InvestigationState] = asyncio.Queue()
_worker_task: asyncio.Task | None = None


async def enqueue(state: InvestigationState) -> None:
    await _queue.put(state)


def _set_status(job_id: str, status: str, error: str | None = None) -> None:
    db = SessionLocal()
    try:
        job = db.get(ReportJob, job_id)
        if job is None:
            return
        job.status = status
        job.error = error
        db.commit()
    finally:
        db.close()


async def _worker() -> None:
    while True:
        state = await _queue.get()
        job_id = state["job_id"]
        try:
            _set_status(job_id, "running")
            result = await asyncio.to_thread(compiled_graph.invoke, state)
            if result.get("ingest_ok"):
                _set_status(job_id, "completed")
            else:
                _set_status(job_id, "failed", result.get("error") or "unknown graph failure")
        except Exception as exc:  # noqa: BLE001 - job worker must never die on a bad job
            logger.exception("job %s failed", job_id)
            _set_status(job_id, "failed", str(exc))
        finally:
            _queue.task_done()


def start_worker() -> None:
    global _worker_task
    if _worker_task is None:
        _worker_task = asyncio.get_event_loop().create_task(_worker())


def stop_worker() -> None:
    global _worker_task
    if _worker_task is not None:
        _worker_task.cancel()
        _worker_task = None
