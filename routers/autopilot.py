"""Autopilot endpoints (/auto/*) — an endless hyperparameter search that runs until stopped.

Unlike /tune (whose worker lives and dies with the browser's SSE connection), the
autopilot worker is owned by the SERVER: it keeps running if the page is closed,
and any page can re-attach to it (GET /auto/stream replays recent events and then
follows live ones). It shells out to autotune_worker.py in .venv-automl.

  POST /auto/start   start (or resume) the endless search
  POST /auto/stop    ask it to stop (graceful; escalates to a kill after a grace period)
  GET  /auto/status  current state: running?, round, best model so far
  GET  /auto/stream  SSE: recent events replayed, then live; ends after "closed"
  GET  /auto/best    best model found so far for a dataset (works while stopped too)

The best model is always saved as models/autopilot_best__<clean_file_id>.joblib in
the same bundle format the inference worker already loads, replaced atomically
whenever a better model is found — so Inference can point at that one filename and
always get "the current best".
"""
import asyncio
import json
import re
import tempfile
import time
import uuid
from collections import deque
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from common import (
    AUTOML_VENV_PYTHON, AUTOTUNE_WORKER_SCRIPT, MODELS_DIR, PREPARED_DIR, _ID_COL_RE,
)

router = APIRouter()

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
MAX_CRASHES = 5        # consecutive worker crashes before giving up
STOP_GRACE_S = 60      # after a stop request, kill the worker if it hasn't exited by then


class AutoRequest(BaseModel):
    clean_file_id: str
    columns: list[str]
    label: str
    label_kind: str            # "numeric" -> regression, "categorical" -> classification
    models: list[str] = []     # empty -> every model autopilot knows a search space for
    fold: int = 3              # CV folds inside each trial (cheap; the hold-out decides winners)
    fresh: bool = False        # True -> discard saved progress/best model and start from round 0


class AutoJob:
    def __init__(self, req: AutoRequest):
        self.id = str(uuid.uuid4())
        self.req = req
        self.status = "starting"   # starting | running | stopping | stopped | error
        self.started_at = time.time()
        self.events: deque = deque(maxlen=4000)   # (seq, event, data)
        self.seq = 0
        self.cond = asyncio.Condition()
        self.proc: asyncio.subprocess.Process | None = None
        self.task: asyncio.Task | None = None
        self.stop_requested = False
        self.round_info: dict | None = None
        self.best: dict | None = None
        self.crashes = 0
        self.last_error: str | None = None
        self.finished = False
        self.stop_file = Path(tempfile.gettempdir()) / f"kaggler_autopilot_stop_{self.id}"
        self.stderr_tail: deque = deque(maxlen=40)

    async def push(self, event: str, data: dict) -> None:
        async with self.cond:
            self.seq += 1
            self.events.append((self.seq, event, data))
            self.cond.notify_all()

    def summary(self) -> dict:
        return {
            "job_id": self.id, "status": self.status, "started_at": self.started_at,
            "round": self.round_info, "best": self.best, "crashes": self.crashes,
            "last_error": self.last_error, "clean_file_id": self.req.clean_file_id,
            "label": self.req.label, "label_kind": self.req.label_kind,
        }


_job: AutoJob | None = None


def _paths(clean_id: str) -> tuple[Path, Path]:
    return (MODELS_DIR / f"autopilot_best__{clean_id}.joblib",
            MODELS_DIR / f"autopilot_state__{clean_id}.json")


def _read_saved_best(clean_id: str) -> dict | None:
    _, state_path = _paths(clean_id)
    try:
        return json.loads(state_path.read_text()).get("best")
    except Exception:
        return None


async def _drain(stream, sink: deque) -> None:
    """Keep reading stderr so a chatty worker can never fill the pipe and block."""
    while True:
        line = await stream.readline()
        if not line:
            return
        sink.append(line.decode("utf-8", errors="replace").rstrip())


async def _run(job: AutoJob, feature_cols: list[str]) -> None:
    req = job.req
    best_path, state_path = _paths(req.clean_file_id)
    args = {
        "csv_path": str(PREPARED_DIR / f"{req.clean_file_id}__clean.csv"),
        "feature_cols": feature_cols,
        "label": req.label,
        "label_kind": req.label_kind,
        "fold": req.fold,
        "models": req.models,
        "models_dir": str(MODELS_DIR),
        "best_path": str(best_path),
        "state_path": str(state_path),
        "stop_file": str(job.stop_file),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(args, f)
        args_path = f.name

    try:
        while True:
            t_start = time.time()
            got_stopped = False
            worker_error: str | None = None
            job.stderr_tail.clear()
            job.proc = await asyncio.create_subprocess_exec(
                str(AUTOML_VENV_PYTHON), str(AUTOTUNE_WORKER_SCRIPT), args_path,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            drain = asyncio.create_task(_drain(job.proc.stderr, job.stderr_tail))
            while True:
                line = await job.proc.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                event = msg.pop("event", "message")
                if event == "error":      # worker-level failure; the supervisor decides what happens next
                    worker_error = msg.get("detail")
                    event = "worker_error"
                elif event == "stopped":
                    got_stopped = True
                elif event == "round_start":
                    job.round_info = msg
                elif event == "new_best":
                    job.best = msg
                if job.status == "starting" and event != "worker_error":
                    job.status = "running"
                await job.push(event, msg)
            rc = await job.proc.wait()
            await drain

            if job.stop_requested or got_stopped:
                job.status = "stopped"
                break

            # Unexpected exit (OOM-kill, bug, ...): restart and resume from the saved state.
            if time.time() - t_start > 600:
                job.crashes = 0          # it ran a long while; this is a fresh failure, not a crash loop
            job.crashes += 1
            detail = worker_error or "\n".join(list(job.stderr_tail)[-8:]) or f"Worker exited with code {rc}"
            job.last_error = detail[-1500:]
            if job.crashes > MAX_CRASHES:
                job.status = "error"
                await job.push("error", {"detail": f"Autopilot gave up after {MAX_CRASHES} crashes. Last error: {job.last_error}"})
                break
            delay = min(60, 5 * job.crashes)
            await job.push("crashed", {"detail": job.last_error, "attempt": job.crashes, "restart_in_s": delay})
            for _ in range(delay):       # interruptible sleep so Stop works during backoff
                if job.stop_requested:
                    break
                await asyncio.sleep(1)
            if job.stop_requested:
                job.status = "stopped"
                break
    finally:
        job.finished = True
        job.proc = None
        Path(args_path).unlink(missing_ok=True)
        job.stop_file.unlink(missing_ok=True)
        await job.push("closed", {"status": job.status, "best": job.best or _read_saved_best(req.clean_file_id)})


@router.post("/auto/start")
async def auto_start(req: AutoRequest):
    global _job
    if _job is not None and not _job.finished:
        raise HTTPException(status_code=409, detail="Autopilot is already running — stop it first")
    if not _SAFE_ID.match(req.clean_file_id):
        raise HTTPException(status_code=400, detail="Invalid clean_file_id")
    if not (PREPARED_DIR / f"{req.clean_file_id}__clean.csv").exists():
        raise HTTPException(status_code=404, detail="Clean file not found")
    if req.label not in req.columns:
        raise HTTPException(status_code=400, detail="Label is not in columns")
    if req.label_kind not in ("categorical", "numeric"):
        raise HTTPException(status_code=400, detail="label_kind must be 'categorical' or 'numeric'")
    if not AUTOML_VENV_PYTHON.exists():
        raise HTTPException(status_code=500, detail="AutoML environment not installed on server (.venv-automl missing)")
    if not 2 <= req.fold <= 10:
        raise HTTPException(status_code=400, detail="fold must be between 2 and 10")

    excluded = [c for c in req.columns if c != req.label and _ID_COL_RE.search(c.strip())]
    feature_cols = [c for c in req.columns if c != req.label and c not in excluded]
    if not feature_cols:
        raise HTTPException(status_code=400, detail="No feature columns to train on")

    best_path, state_path = _paths(req.clean_file_id)
    if req.fresh:
        best_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)

    job = AutoJob(req)
    _job = job
    await job.push("start", {"label": req.label, "feature_columns": feature_cols,
                             "excluded_id_columns": excluded, "resumed": state_path.exists()})
    job.task = asyncio.create_task(_run(job, feature_cols))
    return JSONResponse({"job_id": job.id, "status": job.status})


async def _escalate_kill(job: AutoJob) -> None:
    await asyncio.sleep(STOP_GRACE_S)
    proc = job.proc
    if proc is not None and proc.returncode is None:
        # Safe: the best model is replaced atomically and progress is saved after every
        # model; the interrupted round simply runs again on the next start.
        proc.kill()


@router.post("/auto/stop")
async def auto_stop():
    job = _job
    if job is None or job.finished:
        return JSONResponse({"status": "idle"})
    job.stop_requested = True
    job.status = "stopping"
    job.stop_file.touch()
    asyncio.create_task(_escalate_kill(job))
    await job.push("stopping", {"grace_s": STOP_GRACE_S})
    return JSONResponse({"status": "stopping"})


@router.get("/auto/status")
async def auto_status(clean_file_id: str = ""):
    if _job is not None and (not clean_file_id or _job.req.clean_file_id == clean_file_id):
        return JSONResponse({"active": not _job.finished, **_job.summary()})
    best = _read_saved_best(clean_file_id) if clean_file_id and _SAFE_ID.match(clean_file_id) else None
    return JSONResponse({"active": False, "status": "idle", "best": best})


@router.get("/auto/best")
async def auto_best(clean_file_id: str):
    if not _SAFE_ID.match(clean_file_id):
        raise HTTPException(status_code=400, detail="Invalid clean_file_id")
    best_path, _ = _paths(clean_file_id)
    best = _read_saved_best(clean_file_id)
    if best is None or not best_path.exists():
        raise HTTPException(status_code=404, detail="Autopilot has not found a model for this dataset yet")
    return JSONResponse(best)


@router.get("/auto/stream")
async def auto_stream(request: Request, after: int = 0):
    job = _job
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    if job is None:
        async def _idle():
            yield "event: closed\ndata: " + json.dumps({"status": "idle"}) + "\n\n"
        return StreamingResponse(_idle(), media_type="text/event-stream", headers=headers)

    try:
        last = max(after, int(request.headers.get("last-event-id") or 0))
    except ValueError:
        last = after

    async def gen():
        nonlocal last
        while True:
            for seq, event, data in [e for e in job.events if e[0] > last]:
                last = seq
                yield f"id: {seq}\nevent: {event}\ndata: {json.dumps(data)}\n\n"
                if event == "closed":
                    return
            ping = False
            async with job.cond:
                if job.seq <= last:
                    try:
                        await asyncio.wait_for(job.cond.wait(), timeout=15)
                    except asyncio.TimeoutError:
                        ping = True
            if ping:
                yield ": ping\n\n"
            if await request.is_disconnected():
                return   # only this viewer leaves; the autopilot keeps running

    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


@router.on_event("shutdown")
async def _shutdown() -> None:
    job = _job
    if job is not None and job.proc is not None and job.proc.returncode is None:
        job.stop_requested = True
        job.stop_file.touch()
        job.proc.kill()
