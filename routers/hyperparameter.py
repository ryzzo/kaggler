"""Hyperparameter search endpoints (/tune/*) — shells out to tune_worker.py in .venv-automl.

Runs an Optuna search per selected model using a search space fully specified by
the caller (which parameters, their ranges/choices) — that's what lets the
Hyperparameter Search page offer real control over what gets tuned. Search runs
on the 10% sample; the best-found hyperparameters get one final fit on the full
dataset.
"""
import asyncio
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from common import (
    AUTOML_VENV_PYTHON, MODELS_DIR, PREPARED_DIR, TUNE_WORKER_SCRIPT,
    _ID_COL_RE, _sse,
)

router = APIRouter()


class ModelTuneSpec(BaseModel):
    model_id: str
    n_trials: int = 30
    search_space: dict[str, dict[str, Any]] = {}


class TuneRequest(BaseModel):
    clean_file_id: str
    columns: list[str]
    label: str
    label_kind: str  # "numeric" -> regression, "categorical" -> classification
    models: list[ModelTuneSpec]
    sample_frac: float = 0.1
    fold: int = 5


async def _tune_generator(req: TuneRequest):
    path = PREPARED_DIR / f"{req.clean_file_id}__clean.csv"
    if not path.exists():
        yield _sse("error", {"detail": "Clean file not found", "status": 404})
        return
    if req.label not in req.columns:
        yield _sse("error", {"detail": "Label is not in columns", "status": 400})
        return
    if req.label_kind not in ("categorical", "numeric"):
        yield _sse("error", {"detail": "label_kind must be 'categorical' or 'numeric'", "status": 400})
        return
    if not req.models:
        yield _sse("error", {"detail": "No models selected to tune", "status": 400})
        return
    if not AUTOML_VENV_PYTHON.exists():
        yield _sse("error", {"detail": "AutoML environment not installed on server (.venv-automl missing)", "status": 500})
        return

    excluded_id_cols = [c for c in req.columns if c != req.label and _ID_COL_RE.search(c.strip())]
    feature_cols = [c for c in req.columns if c != req.label and c not in excluded_id_cols]
    if not feature_cols:
        yield _sse("error", {"detail": "No feature columns to train on", "status": 400})
        return

    run_id = str(uuid.uuid4())
    args = {
        "csv_path": str(path),
        "feature_cols": feature_cols,
        "label": req.label,
        "label_kind": req.label_kind,
        "sample_frac": req.sample_frac,
        "fold": req.fold,
        "models": [m.model_dump() for m in req.models],
        "models_dir": str(MODELS_DIR),
        "run_id": run_id,
    }

    yield _sse("start", {
        "label": req.label,
        "feature_columns": feature_cols,
        "excluded_id_columns": excluded_id_cols,
        "sample_frac": req.sample_frac,
        "models": [m.model_id for m in req.models],
    })

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(args, f)
        args_path = f.name

    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            str(AUTOML_VENV_PYTHON), str(TUNE_WORKER_SCRIPT), args_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        saw_error = False
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            event = msg.pop("event", "message")
            if event == "error":
                saw_error = True
            yield _sse(event, msg)

        stderr_tail = (await proc.stderr.read()).decode("utf-8", errors="replace")
        returncode = await proc.wait()
        if returncode != 0 and not saw_error:
            yield _sse("error", {"detail": stderr_tail[-2000:] or f"Worker exited with code {returncode}", "status": 500})
    finally:
        # If the client disconnects mid-run (e.g. navigates away), don't leave the
        # worker running orphaned in the background — it can be a multi-minute fit.
        if proc is not None and proc.returncode is None:
            proc.kill()
        Path(args_path).unlink(missing_ok=True)


_job_store: dict[str, TuneRequest] = {}


@router.post("/tune/init")
async def tune_init(req: TuneRequest):
    """Validate and store hyperparameter search params; return a job_id for the SSE stream."""
    path = PREPARED_DIR / f"{req.clean_file_id}__clean.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Clean file not found")
    if req.label not in req.columns:
        raise HTTPException(status_code=400, detail="Label is not in columns")
    if req.label_kind not in ("categorical", "numeric"):
        raise HTTPException(status_code=400, detail="label_kind must be 'categorical' or 'numeric'")
    if not req.models:
        raise HTTPException(status_code=400, detail="No models selected to tune")
    job_id = str(uuid.uuid4())
    _job_store[job_id] = req
    return JSONResponse({"job_id": job_id})


@router.get("/tune/stream/{job_id}")
async def tune_stream(job_id: str):
    """SSE endpoint consumed by EventSource — streams hyperparameter search progress as it completes."""
    req = _job_store.pop(job_id, None)
    if req is None:
        async def _not_found():
            yield _sse("error", {"detail": "Job not found or already consumed", "status": 404})
        return StreamingResponse(_not_found(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return StreamingResponse(
        _tune_generator(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
