"""PyCaret AutoML endpoints (/automl/*) — shells out to automl_worker.py in .venv-automl."""
import asyncio
import json
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from common import (
    AUTOML_VENV_PYTHON, AUTOML_WORKER_SCRIPT, MODELS_DIR, PREPARED_DIR,
    _ID_COL_RE, _sse,
)

router = APIRouter()


class AutoMLRequest(BaseModel):
    clean_file_id: str
    columns: list[str]
    label: str
    label_kind: str  # "numeric" -> regression, "categorical" -> classification
    sample_frac: float = 0.1


async def _automl_generator(req: AutoMLRequest):
    path = PREPARED_DIR / f"{req.clean_file_id}__clean.csv"
    if not path.exists():
        yield _sse("error", {"detail": "Clean file not found", "status": 404})
        return
    if req.label not in req.columns:
        yield _sse("error", {"detail": "Label is not in columns", "status": 400})
        return
    if req.label_kind != "categorical":
        yield _sse("error", {"detail": "AutoML currently supports classification labels only", "status": 400})
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
        "models_dir": str(MODELS_DIR),
        "run_id": run_id,
    }

    yield _sse("start", {
        "label": req.label,
        "feature_columns": feature_cols,
        "excluded_id_columns": excluded_id_cols,
        "sample_frac": req.sample_frac,
    })

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(args, f)
        args_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            str(AUTOML_VENV_PYTHON), str(AUTOML_WORKER_SCRIPT), args_path,
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
        Path(args_path).unlink(missing_ok=True)


_automl_job_store: dict[str, AutoMLRequest] = {}


@router.post("/automl/init")
async def automl_init(req: AutoMLRequest):
    """Validate and store AutoML params; return a job_id for the SSE stream."""
    path = PREPARED_DIR / f"{req.clean_file_id}__clean.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Clean file not found")
    if req.label not in req.columns:
        raise HTTPException(status_code=400, detail="Label is not in columns")
    if req.label_kind != "categorical":
        raise HTTPException(status_code=400, detail="AutoML currently supports classification labels only")
    job_id = str(uuid.uuid4())
    _automl_job_store[job_id] = req
    return JSONResponse({"job_id": job_id})


@router.get("/automl/stream/{job_id}")
async def automl_stream(job_id: str):
    """SSE endpoint consumed by EventSource — streams PyCaret AutoML progress as it completes."""
    req = _automl_job_store.pop(job_id, None)
    if req is None:
        async def _not_found():
            yield _sse("error", {"detail": "Job not found or already consumed", "status": 404})
        return StreamingResponse(_not_found(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return StreamingResponse(
        _automl_generator(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
