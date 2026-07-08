"""GradientBoosting/LightGBM/CatBoost training + voting ensembles (/train/*)."""
import asyncio
import csv
import io
import math
import time
import uuid
from collections import Counter
from itertools import combinations
from typing import Any

import joblib
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from lightgbm import LGBMClassifier, LGBMRegressor
from pydantic import BaseModel
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import train_test_split

from common import MODELS_DIR, PREPARED_DIR, _ID_COL_RE, _sse
from routers.resources import GPU_AVAILABLE

router = APIRouter()


class TrainRequest(BaseModel):
    prepared_file_id: str
    columns: list[str]
    label: str
    label_kind: str  # "numeric" -> regression, "categorical" -> classification


async def _train_generator(req: TrainRequest):
    path = PREPARED_DIR / f"{req.prepared_file_id}.csv"
    if not path.exists():
        yield _sse("error", {"detail": "Prepared file not found", "status": 404})
        return
    if req.label not in req.columns:
        yield _sse("error", {"detail": "Label is not in columns", "status": 400})
        return

    excluded_id_cols = [c for c in req.columns if c != req.label and _ID_COL_RE.search(c.strip())]
    feature_cols = [c for c in req.columns if c != req.label and c not in excluded_id_cols]
    if not feature_cols:
        yield _sse("error", {"detail": "No feature columns to train on", "status": 400})
        return
    if req.label_kind not in ("numeric", "categorical"):
        yield _sse("error", {"detail": "label_kind must be 'numeric' or 'categorical'", "status": 400})
        return

    text = path.read_text(encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    if len(rows) < 5:
        yield _sse("error", {"detail": "Not enough rows to train/test split", "status": 400})
        return

    X = np.array([[float(r[c]) for c in feature_cols] for r in rows])
    is_classification = req.label_kind == "categorical"
    y_raw = np.array([float(r[req.label]) for r in rows])
    y = y_raw.astype(int) if is_classification else y_raw

    stratify = None
    if is_classification:
        counts = Counter(y.tolist())
        if len(counts) > 1 and min(counts.values()) >= 2:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    lgbm_extra = {"device": "gpu"} if GPU_AVAILABLE else {}
    catboost_extra = {"task_type": "GPU"} if GPU_AVAILABLE else {}

    model_defs = (
        [
            ("gradient_boosting", GradientBoostingClassifier(random_state=42)),
            ("lightgbm",          LGBMClassifier(random_state=42, verbose=-1, **lgbm_extra)),
            ("catboost",          CatBoostClassifier(random_state=42, verbose=False, **catboost_extra)),
        ] if is_classification else [
            ("gradient_boosting", GradientBoostingRegressor(random_state=42)),
            ("lightgbm",          LGBMRegressor(random_state=42, verbose=-1, **lgbm_extra)),
            ("catboost",          CatBoostRegressor(random_state=42, verbose=False, **catboost_extra)),
        ]
    )
    total = len(model_defs)

    # emit setup info so the frontend can build the table skeleton
    yield _sse("start", {
        "task_type": "classification" if is_classification else "regression",
        "label": req.label,
        "feature_columns": feature_cols,
        "excluded_id_columns": excluded_id_cols,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "models": [n for n, _ in model_defs],
        "total": total,
    })

    results: dict[str, Any] = {}
    models: dict[str, Any] = {}
    test_preds: dict[str, np.ndarray] = {}
    test_probas: dict[str, np.ndarray] = {}
    classes_: np.ndarray | None = None

    for idx, (name, model) in enumerate(model_defs):
        yield _sse("training", {"model": name, "index": idx, "total": total})
        t0 = time.perf_counter()
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, model.fit, X_train, y_train)
            preds = np.asarray(model.predict(X_test)).ravel()
            elapsed = round(time.perf_counter() - t0, 2)
            test_preds[name] = preds
            models[name] = model
            if is_classification:
                metrics = {
                    "accuracy":    round(float(accuracy_score(y_test, preds)), 4),
                    "f1_weighted": round(float(f1_score(y_test, preds, average="weighted")), 4),
                }
                proba = model.predict_proba(X_test)
                if classes_ is None:
                    classes_ = np.asarray(model.classes_)
                if np.array_equal(np.asarray(model.classes_), classes_):
                    test_probas[name] = np.asarray(proba)
            else:
                metrics = {
                    "rmse": round(float(math.sqrt(mean_squared_error(y_test, preds))), 4),
                    "mae":  round(float(mean_absolute_error(y_test, preds)), 4),
                    "r2":   round(float(r2_score(y_test, preds)), 4),
                }
            results[name] = metrics
            yield _sse("result", {"model": name, "index": idx, "metrics": metrics, "elapsed_s": elapsed})
        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 2)
            results[name] = {"error": str(exc)}
            yield _sse("result", {"model": name, "index": idx, "error": str(exc), "elapsed_s": elapsed})

    # ── Voting ensembles ──
    ok_names = [n for n, _ in model_defs if n in test_preds]
    ensemble_results = []
    for r in (2, 3):
        for combo in combinations(ok_names, r):
            if is_classification:
                if all(n in test_probas for n in combo):
                    avg_proba = np.mean([test_probas[n] for n in combo], axis=0)
                    soft_pred = classes_[np.argmax(avg_proba, axis=1)]
                    ensemble_results.append({
                        "models": list(combo), "voting": "soft",
                        "accuracy":    round(float(accuracy_score(y_test, soft_pred)), 4),
                        "f1_weighted": round(float(f1_score(y_test, soft_pred, average="weighted")), 4),
                    })
            else:
                avg_pred = np.mean([test_preds[n] for n in combo], axis=0)
                ensemble_results.append({
                    "models": list(combo), "voting": "average",
                    "rmse": round(float(math.sqrt(mean_squared_error(y_test, avg_pred))), 4),
                    "mae":  round(float(mean_absolute_error(y_test, avg_pred)), 4),
                    "r2":   round(float(r2_score(y_test, avg_pred)), 4),
                })

    yield _sse("ensembles", {"ensemble_results": ensemble_results})

    # ── Save top-2 models ──
    rank_metric = "accuracy" if is_classification else "r2"
    ranked = sorted(
        ((name, results[name][rank_metric]) for name in results if "error" not in results[name] and rank_metric in results[name]),
        key=lambda item: item[1], reverse=True,
    )
    saved_models = []
    for name, score in ranked[:2]:
        bundle = {
            "model": models[name],
            "model_name": name,
            "task_type": "classification" if is_classification else "regression",
            "feature_columns": feature_cols,
            "label": req.label,
            "label_kind": req.label_kind,
            "metric": rank_metric,
            "score": score,
        }
        filename = f"{req.prepared_file_id}__{name}.joblib"
        joblib.dump(bundle, MODELS_DIR / filename)
        saved_models.append({"model": name, "metric": rank_metric, "score": score, "filename": filename})

    yield _sse("saved", {"saved_models": saved_models})
    yield _sse("done", {
        "task_type": "classification" if is_classification else "regression",
        "label": req.label,
        "feature_columns": feature_cols,
        "excluded_id_columns": excluded_id_cols,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "results": results,
        "ensemble_results": ensemble_results,
        "saved_models": saved_models,
    })


_job_store: dict[str, TrainRequest] = {}


@router.post("/train/init")
async def train_init(req: TrainRequest):
    """Validate and store training params; return a job_id for the SSE stream."""
    path = PREPARED_DIR / f"{req.prepared_file_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Prepared file not found")
    if req.label not in req.columns:
        raise HTTPException(status_code=400, detail="Label is not in columns")
    if req.label_kind not in ("numeric", "categorical"):
        raise HTTPException(status_code=400, detail="label_kind must be 'numeric' or 'categorical'")
    job_id = str(uuid.uuid4())
    _job_store[job_id] = req
    return JSONResponse({"job_id": job_id})


@router.get("/train/stream/{job_id}")
async def train_stream(job_id: str):
    """SSE endpoint consumed by EventSource — streams model results as they complete."""
    req = _job_store.pop(job_id, None)
    if req is None:
        async def _not_found():
            yield _sse("error", {"detail": "Job not found or already consumed", "status": 404})
        return StreamingResponse(_not_found(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return StreamingResponse(
        _train_generator(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
