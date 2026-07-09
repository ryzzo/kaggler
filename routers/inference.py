"""Batch inference against a saved model (/infer) and its download endpoints.

Also /infer/pycaret/* — real inference for a model selected on the Hyperparameter
Search page, via infer_worker.py in the isolated .venv-automl (same subprocess
pattern as /train and /tune): it reconstructs PyCaret's preprocessing pipeline
on the full clean training data, reuses a saved fitted model if one exists or
fits fresh otherwise, and predicts on the uploaded test CSV through that
pipeline.
"""
import asyncio
import csv
import io
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from common import (
    AUTOML_VENV_PYTHON, INFER_WORKER_SCRIPT, MODELS_DIR, PREDICTIONS_DIR,
    PREPARED_DIR, _ID_COL_RE, _sse,
)

router = APIRouter()


def _apply_imputation_params(
    rows: list[dict[str, str]],
    imputation_params: dict[str, dict],
) -> int:
    """Apply saved imputation params to test rows in-place. Returns count of filled cells."""
    filled = 0

    def _is_blank(v: str) -> bool:
        return v.strip() == ""

    for col, params in imputation_params.items():
        method = params.get("method", "skipped")
        if method == "skipped":
            continue
        for r in rows:
            if col not in r or not _is_blank(r[col]):
                continue
            if method in ("median", "median_fallback"):
                r[col] = str(params["value"])
                filled += 1
            elif method in ("mode", "mode_fallback"):
                r[col] = str(params["value"])
                filled += 1
            elif method == "knn":
                feat_cols: list[str] = params["feature_cols"]
                col_means: dict[str, float] = params["col_means"]
                knn_model = params["model"]
                x = [float(r[c]) if c in r and not _is_blank(r[c]) else col_means.get(c, 0.0)
                     for c in feat_cols]
                pred = knn_model.predict([x])[0]
                r[col] = str(round(float(pred), 6)) if isinstance(pred, (int, float)) else str(pred)
                filled += 1
    return filled


@router.post("/infer")
async def infer(
    file: UploadFile = File(...),
    model_filename: str = Form(...),
    feature_columns: str = Form(...),
    label: str = Form(...),
    task_type: str = Form(...),
    encoding_info: str = Form(...),
    scaling_info: str = Form(...),
    imputation_params_id: str = Form(default=""),
):
    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    bundle = joblib.load(model_path)
    model = bundle["model"]

    try:
        feats: list[str] = json.loads(feature_columns)
        enc: dict[str, dict[str, int]] = json.loads(encoding_info)
        scale: dict[str, dict[str, float]] = json.loads(scaling_info)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in form fields")

    # Load imputation params if provided
    imp_params: dict[str, dict] = {}
    if imputation_params_id:
        imp_path = PREPARED_DIR / f"{imputation_params_id}__imputation.joblib"
        if imp_path.exists():
            imp_params = joblib.load(imp_path)

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    text = content.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="No rows in file")

    fieldnames = reader.fieldnames or []
    missing_cols = set(feats) - set(fieldnames)
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Test file is missing required columns: {sorted(missing_cols)}")

    # Apply same imputation used during training
    imputed_cells = _apply_imputation_params(rows, imp_params) if imp_params else 0

    unseen_categories = 0
    X = []
    for r in rows:
        vec = []
        for c in feats:
            raw_v = r.get(c, "")
            if c in enc:
                mapping = enc[c]
                key = raw_v if raw_v.strip() != "" else "__missing__"
                if key in mapping:
                    code = mapping[key]
                else:
                    code = -1
                    unseen_categories += 1
                vec.append(float(code))
            else:
                stats = scale.get(c)
                mean = stats["mean"] if stats else 0.0
                try:
                    num = float(raw_v) if raw_v.strip() != "" else mean
                except ValueError:
                    num = mean
                if stats:
                    std = stats["std"] or 1.0
                    vec.append((num - mean) / std)
                else:
                    vec.append(num)
        X.append(vec)

    X = np.array(X)
    raw_preds = model.predict(X)

    if task_type == "classification" and label in enc:
        inv_map = {v: k for k, v in enc[label].items()}
        predictions = [inv_map.get(int(round(float(p))), str(p)) for p in raw_preds]
    elif task_type == "regression" and label in scale:
        mean = scale[label]["mean"]
        std = scale[label]["std"]
        predictions = [round(float(p) * std + mean, 6) for p in raw_preds]
    else:
        predictions = [float(p) for p in raw_preds]

    pred_col = f"{label}_prediction"
    out_cols = fieldnames + [pred_col]
    pred_id = str(uuid.uuid4())

    # Full predictions file (all input columns + prediction)
    save_path = PREDICTIONS_DIR / f"{pred_id}.csv"
    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(out_cols)
        for r, p in zip(rows, predictions):
            writer.writerow([r.get(c, "") for c in fieldnames] + [p])

    # Kaggle submission file: id column (or row index) + label column
    id_col = next((c for c in fieldnames if _ID_COL_RE.match(c)), None)
    sub_path = PREDICTIONS_DIR / f"{pred_id}_submission.csv"
    with sub_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id" if id_col is None else id_col, label])
        for i, (r, p) in enumerate(zip(rows, predictions)):
            id_val = str(i) if id_col is None else r.get(id_col, str(i))
            writer.writerow([id_val, p])

    preview = [
        {**{c: r.get(c, "") for c in fieldnames}, pred_col: p}
        for r, p in list(zip(rows, predictions))[:10]
    ]

    return JSONResponse({
        "row_count": len(rows),
        "model_used": bundle.get("model_name"),
        "prediction_column": pred_col,
        "unseen_categories": unseen_categories,
        "imputed_cells": imputed_cells,
        "columns": out_cols,
        "preview": preview,
        "download_url": f"/download/predictions/{pred_id}",
        "submission_url": f"/download/submission/{pred_id}",
        "submission_id_col": id_col or "id (row index)",
    })


class PycaretInferRequest(BaseModel):
    clean_file_id: str
    columns: list[str]
    label: str
    label_kind: str
    model_id: str
    best_params: dict[str, Any] = {}
    model_filename: str = ""


_pycaret_infer_jobs: dict[str, dict] = {}


@router.post("/infer/pycaret/init")
async def infer_pycaret_init(
    file: UploadFile = File(...),
    clean_file_id: str = Form(...),
    columns: str = Form(...),
    label: str = Form(...),
    label_kind: str = Form(...),
    model_id: str = Form(...),
    best_params: str = Form(default="{}"),
    model_filename: str = Form(default=""),
):
    """Validate, stash the uploaded test CSV to disk, and return a job_id for the SSE stream."""
    clean_path = PREPARED_DIR / f"{clean_file_id}__clean.csv"
    if not clean_path.exists():
        raise HTTPException(status_code=404, detail="Clean training file not found")
    if label_kind not in ("categorical", "numeric"):
        raise HTTPException(status_code=400, detail="label_kind must be 'categorical' or 'numeric'")
    if not AUTOML_VENV_PYTHON.exists():
        raise HTTPException(status_code=500, detail="AutoML environment not installed on server (.venv-automl missing)")

    try:
        cols: list[str] = json.loads(columns)
        params: dict = json.loads(best_params)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in form fields")

    # Derive feature columns the same way /train/init and /tune/init do — never
    # trust a client-supplied feature list directly, so an id-like column can't
    # sneak in as a "feature" here when it was excluded during training (that
    # mismatch is exactly what causes LightGBM's "number of features" error).
    excluded_id_cols = [c for c in cols if c != label and _ID_COL_RE.search(c.strip())]
    feats = [c for c in cols if c != label and c not in excluded_id_cols]
    if not feats:
        raise HTTPException(status_code=400, detail="No feature columns to run inference on")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as f:
        f.write(content)
        test_csv_path = f.name

    model_path = ""
    if model_filename:
        candidate = MODELS_DIR / model_filename
        if candidate.exists():
            model_path = str(candidate)

    job_id = str(uuid.uuid4())
    _pycaret_infer_jobs[job_id] = {
        "clean_csv_path": str(clean_path),
        "test_csv_path": test_csv_path,
        "feature_cols": feats,
        "label": label,
        "label_kind": label_kind,
        "model_id": model_id,
        "best_params": params,
        "model_path": model_path,
    }
    return JSONResponse({"job_id": job_id})


async def _pycaret_infer_generator(job: dict):
    pred_id = str(uuid.uuid4())
    args = {
        **job,
        "predictions_dir": str(PREDICTIONS_DIR),
        "pred_id": pred_id,
    }
    test_csv_path = job["test_csv_path"]

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(args, f)
        args_path = f.name

    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            str(AUTOML_VENV_PYTHON), str(INFER_WORKER_SCRIPT), args_path,
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
            if event == "done":
                msg["download_url"] = f"/download/predictions/{pred_id}"
                msg["submission_url"] = f"/download/submission/{pred_id}"
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
        Path(test_csv_path).unlink(missing_ok=True)


@router.get("/infer/pycaret/stream/{job_id}")
async def infer_pycaret_stream(job_id: str):
    """SSE endpoint consumed by EventSource — streams PyCaret inference progress as it completes."""
    job = _pycaret_infer_jobs.pop(job_id, None)
    if job is None:
        async def _not_found():
            yield _sse("error", {"detail": "Job not found or already consumed", "status": 404})
        return StreamingResponse(_not_found(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return StreamingResponse(
        _pycaret_infer_generator(job),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/download/predictions/{pred_id}")
def download_predictions(pred_id: str):
    path = PREDICTIONS_DIR / f"{pred_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/csv", filename="predictions.csv")


@router.get("/download/submission/{pred_id}")
def download_submission(pred_id: str):
    path = PREDICTIONS_DIR / f"{pred_id}_submission.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/csv", filename="submission.csv")
