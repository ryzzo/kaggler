import asyncio
import io
import csv
import json
import math
import re
import subprocess
import time
import uuid
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import uvicorn
import numpy as np
import joblib
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# ── GPU detection ────────────────────────────────────────────────────────────
def _detect_gpu() -> bool:
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False

GPU_AVAILABLE = _detect_gpu()
print(f"[startup] GPU available: {GPU_AVAILABLE}")

# ── Resource helpers (no psutil — pure /proc) ─────────────────────────────
def _read_ram() -> dict:
    """Return total/used/free RAM in MB and percent used."""
    info = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            info[k.strip()] = int(v.split()[0])  # kB
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        used  = total - avail
        return {
            "total_mb": round(total / 1024, 1),
            "used_mb":  round(used  / 1024, 1),
            "free_mb":  round(avail / 1024, 1),
            "percent":  round(used / total * 100, 1) if total else 0,
        }
    except Exception:
        return {}

_cpu_prev: list[int] = []

def _read_cpu() -> float:
    """Estimate CPU % over ~200 ms using /proc/stat."""
    global _cpu_prev
    def _stat():
        line = Path("/proc/stat").read_text().splitlines()[0].split()
        nums = list(map(int, line[1:]))
        idle  = nums[3]
        total = sum(nums)
        return total, idle
    try:
        t1, i1 = _stat()
        time.sleep(0.2)
        t2, i2 = _stat()
        dt, di = t2 - t1, i2 - i1
        return round((1 - di / dt) * 100, 1) if dt else 0.0
    except Exception:
        return 0.0

def _read_gpu() -> dict:
    """Return GPU utilisation and VRAM via nvidia-smi CSV query."""
    if not GPU_AVAILABLE:
        return {}
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return {}
        parts = out.stdout.strip().split(",")
        return {
            "util_pct":   float(parts[0].strip()),
            "vram_used_mb": float(parts[1].strip()),
            "vram_total_mb": float(parts[2].strip()),
            "vram_pct":   round(float(parts[1]) / float(parts[2]) * 100, 1),
        }
    except Exception:
        return {}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

PREPARED_DIR = Path("prepared")
PREPARED_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

PREDICTIONS_DIR = Path("predictions")
PREDICTIONS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="File Processing API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/analysis")
def analysis_page():
    return FileResponse("static/analysis.html")


@app.get("/preparation")
def preparation_page():
    return FileResponse("static/preparation.html")


@app.get("/training")
def training_page():
    return FileResponse("static/training.html")


@app.get("/inference")
def inference_page():
    return FileResponse("static/inference.html")


def process_csv(content: bytes) -> dict[str, Any]:
    text = content.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    return {
        "type": "csv",
        "row_count": len(rows),
        "columns": reader.fieldnames or [],
        "preview": rows[:5],
    }


def process_image(content: bytes, filename: str) -> dict[str, Any]:
    img = Image.open(io.BytesIO(content))
    return {
        "type": "image",
        "format": img.format,
        "mode": img.mode,
        "size": {"width": img.size[0], "height": img.size[1]},
        "filename": filename,
    }


def process_text(content: bytes) -> dict[str, Any]:
    text = content.decode("utf-8", errors="replace")
    lines = text.splitlines()
    words = text.split()
    return {
        "type": "text",
        "line_count": len(lines),
        "word_count": len(words),
        "char_count": len(text),
        "preview": text[:500],
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()

    for old in UPLOAD_DIR.iterdir():
        if old.is_file():
            old.unlink()

    file_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{file_id}{suffix}"
    save_path.write_bytes(content)

    try:
        if suffix == ".csv":
            result = process_csv(content)
        elif suffix in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}:
            result = process_image(content, filename)
        elif suffix in {".txt", ".md", ".log"}:
            result = process_text(content)
        else:
            result = {"type": "binary", "size_bytes": len(content)}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Processing error: {exc}") from exc

    return JSONResponse({
        "file_id": file_id,
        "original_filename": filename,
        "size_bytes": len(content),
        "saved_to": str(save_path),
        "processing": result,
    })


@app.post("/upload/batch")
async def upload_batch(files: list[UploadFile] = File(...)):
    for old in UPLOAD_DIR.iterdir():
        if old.is_file():
            old.unlink()

    results = []
    for file in files:
        content = await file.read()
        filename = file.filename or "unknown"
        suffix = Path(filename).suffix.lower()
        file_id = str(uuid.uuid4())
        save_path = UPLOAD_DIR / f"{file_id}{suffix}"
        save_path.write_bytes(content)

        try:
            if suffix == ".csv":
                proc = process_csv(content)
            elif suffix in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}:
                proc = process_image(content, filename)
            elif suffix in {".txt", ".md", ".log"}:
                proc = process_text(content)
            else:
                proc = {"type": "binary", "size_bytes": len(content)}
        except Exception as exc:
            proc = {"error": str(exc)}

        results.append({
            "file_id": file_id,
            "original_filename": filename,
            "size_bytes": len(content),
            "processing": proc,
        })

    return JSONResponse({"files": results, "count": len(results)})


class ColumnSelection(BaseModel):
    file_id: str
    features: list[str]
    label: str


@app.post("/select-columns")
def select_columns(sel: ColumnSelection):
    matches = list(UPLOAD_DIR.glob(f"{sel.file_id}.csv"))
    if not matches:
        raise HTTPException(status_code=404, detail="File not found")

    text = matches[0].read_text(encoding="utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    all_cols = set(reader.fieldnames or [])

    missing = (set(sel.features) | {sel.label}) - all_cols
    if missing:
        raise HTTPException(status_code=400, detail=f"Unknown columns: {sorted(missing)}")
    if sel.label in sel.features:
        raise HTTPException(status_code=400, detail="Label column cannot also be a feature")

    feature_data = {col: [r[col] for r in rows] for col in sel.features}
    label_data   = [r[sel.label] for r in rows]

    return JSONResponse({
        "file_id": sel.file_id,
        "row_count": len(rows),
        "features": sel.features,
        "label": sel.label,
        "feature_sample": {col: vals[:5] for col, vals in feature_data.items()},
        "label_sample": label_data[:5],
    })


def _to_float(values: list[str]) -> list[float] | None:
    try:
        return [float(v) for v in values if v.strip() != ""]
    except ValueError:
        return None


def _stats(nums: list[float]) -> dict:
    n = len(nums)
    if n == 0:
        return {}
    mean = sum(nums) / n
    variance = sum((x - mean) ** 2 for x in nums) / n
    std = math.sqrt(variance)
    sorted_nums = sorted(nums)
    def percentile(p):
        idx = (n - 1) * p / 100
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return sorted_nums[lo] + (sorted_nums[hi] - sorted_nums[lo]) * (idx - lo)
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(sorted_nums[0], 4),
        "q25": round(percentile(25), 4),
        "median": round(percentile(50), 4),
        "q75": round(percentile(75), 4),
        "max": round(sorted_nums[-1], 4),
        "missing": 0,
    }


def _histogram(nums: list[float], bins: int = 20) -> dict:
    if not nums:
        return {"edges": [], "counts": []}
    lo, hi = min(nums), max(nums)
    if lo == hi:
        return {"edges": [lo, hi], "counts": [len(nums)]}
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in nums:
        idx = min(int((v - lo) / width), bins - 1)
        counts[idx] += 1
    edges = [round(lo + i * width, 4) for i in range(bins + 1)]
    return {"edges": edges, "counts": counts}


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx  = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy  = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return round(cov / (sx * sy), 4)


class AnalyzeRequest(BaseModel):
    file_id: str
    features: list[str]
    label: str


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    matches = list(UPLOAD_DIR.glob(f"{req.file_id}.csv"))
    if not matches:
        raise HTTPException(status_code=404, detail="File not found")

    text = matches[0].read_text(encoding="utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    all_cols = set(reader.fieldnames or [])

    missing_cols = (set(req.features) | {req.label}) - all_cols
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Unknown columns: {sorted(missing_cols)}")

    raw: dict[str, list[str]] = {
        col: [r[col] for r in rows] for col in [*req.features, req.label]
    }

    def _missing_count(values: list[str]) -> int:
        return sum(1 for v in values if v.strip() == "")

    label_nums = _to_float(raw[req.label])
    label_cats = Counter(raw[req.label]) if label_nums is None else None

    feature_analysis = {}
    numeric_cols: dict[str, list[float]] = {}
    for col in req.features:
        missing = _missing_count(raw[col])
        nums = _to_float(raw[col])
        if nums is not None:
            corr = _correlation(nums, label_nums) if label_nums else None
            numeric_cols[col] = nums
            feature_analysis[col] = {
                "kind": "numeric",
                "n_unique": len(set(nums)),
                "missing": missing,
                "stats": _stats(nums),
                "histogram": _histogram(nums),
                "correlation_with_label": corr,
            }
        else:
            counts = Counter(raw[col])
            feature_analysis[col] = {
                "kind": "categorical",
                "value_counts": dict(counts.most_common(20)),
                "n_unique": len(counts),
                "missing": missing,
            }

    label_missing = _missing_count(raw[req.label])
    label_info: dict[str, Any] = {"name": req.label}
    if label_nums is not None:
        label_info["kind"] = "numeric"
        label_info["n_unique"] = len(set(label_nums))
        label_info["missing"] = label_missing
        label_info["stats"] = _stats(label_nums)
        label_info["histogram"] = _histogram(label_nums)
    else:
        label_info["kind"] = "categorical"
        label_info["value_counts"] = dict(label_cats.most_common(20))  # type: ignore[union-attr]
        label_info["n_unique"] = len(label_cats)  # type: ignore[arg-type]
        label_info["missing"] = label_missing

    # pairwise correlation matrix (numeric features only)
    num_keys = list(numeric_cols.keys())
    corr_matrix = {
        a: {b: (1.0 if a == b else _correlation(numeric_cols[a], numeric_cols[b]))
            for b in num_keys}
        for a in num_keys
    }

    # ── Missingness correlation ──────────────────────────────────────────────
    # For each column with ≥1 missing value build a binary missing-indicator vector,
    # then correlate it with (a) every other column's indicator and (b) numeric values.
    all_cols_ordered = [*req.features, req.label]
    miss_indicators: dict[str, list[float]] = {
        col: [1.0 if v.strip() == "" else 0.0 for v in raw[col]]
        for col in all_cols_ordered
        if _missing_count(raw[col]) > 0
    }

    missingness_correlation: dict[str, dict] = {}
    if miss_indicators:
        # correlate each missing indicator with all numeric columns + all other indicators
        for miss_col, indicator in miss_indicators.items():
            row_corr: dict[str, float | None] = {}
            # vs numeric column values
            for num_col, num_vals in numeric_cols.items():
                if num_col == miss_col:
                    continue
                row_corr[num_col] = _correlation(indicator, num_vals)
            # vs other missingness indicators
            for other_col, other_ind in miss_indicators.items():
                if other_col == miss_col:
                    continue
                key = f"{other_col} (missing)"
                row_corr[key] = _correlation(indicator, other_ind)
            # sort by absolute value descending, drop None
            sorted_corr = sorted(
                ((k, v) for k, v in row_corr.items() if v is not None),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            missingness_correlation[miss_col] = {
                "missing_count": _missing_count(raw[miss_col]),
                "missing_pct": round(_missing_count(raw[miss_col]) / len(rows) * 100, 2),
                "top_correlations": [
                    {"column": k, "r": round(v, 4)} for k, v in sorted_corr[:10]
                ],
            }

    return JSONResponse({
        "file_id": req.file_id,
        "row_count": len(rows),
        "features": req.features,
        "label": label_info,
        "feature_analysis": feature_analysis,
        "correlation_matrix": {"columns": num_keys, "values": corr_matrix},
        "missingness_correlation": missingness_correlation,
    })


def _label_encode(values: list[str]) -> tuple[list[int], dict[str, int]]:
    uniques = sorted(set(values))
    mapping = {v: i for i, v in enumerate(uniques)}
    return [mapping[v] for v in values], mapping


def _standard_scale(nums: list[float]) -> tuple[list[float], float, float]:
    mean = sum(nums) / len(nums)
    std = math.sqrt(sum((x - mean) ** 2 for x in nums) / len(nums))
    if std == 0:
        std = 1.0
    return [round((x - mean) / std, 6) for x in nums], round(mean, 6), round(std, 6)


class PrepareRequest(BaseModel):
    file_id: str
    features: list[str]
    label: str
    column_kinds: dict[str, str] = {}


@app.post("/prepare")
def prepare(req: PrepareRequest):
    matches = list(UPLOAD_DIR.glob(f"{req.file_id}.csv"))
    if not matches:
        raise HTTPException(status_code=404, detail="File not found")

    text = matches[0].read_text(encoding="utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    all_cols = set(reader.fieldnames or [])

    missing_cols = (set(req.features) | {req.label}) - all_cols
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Unknown columns: {sorted(missing_cols)}")

    cols = [*req.features, req.label]
    prepared: dict[str, list[float]] = {}
    encoding_info: dict[str, dict[str, int]] = {}
    scaling_info: dict[str, dict[str, float]] = {}

    for col in cols:
        raw_values = [r[col] for r in rows]
        kind = req.column_kinds.get(col) or ("numeric" if _to_float(raw_values) is not None else "categorical")

        if kind == "categorical":
            filled = [v if v.strip() != "" else "__missing__" for v in raw_values]
            encoded, mapping = _label_encode(filled)
            prepared[col] = [float(v) for v in encoded]
            encoding_info[col] = mapping
        else:
            nums = [float(v) if v.strip() != "" else None for v in raw_values]
            present = [v for v in nums if v is not None]
            mean = sum(present) / len(present) if present else 0.0
            filled_nums = [v if v is not None else mean for v in nums]
            scaled, mean_r, std_r = _standard_scale(filled_nums)
            prepared[col] = scaled
            scaling_info[col] = {"mean": mean_r, "std": std_r}

    file_id = str(uuid.uuid4())
    save_path = PREPARED_DIR / f"{file_id}.csv"
    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for i in range(len(rows)):
            writer.writerow([prepared[col][i] for col in cols])

    preview = [{col: prepared[col][i] for col in cols} for i in range(min(10, len(rows)))]

    return JSONResponse({
        "file_id": req.file_id,
        "prepared_file_id": file_id,
        "row_count": len(rows),
        "columns": cols,
        "encoding_info": encoding_info,
        "scaling_info": scaling_info,
        "preview": preview,
        "download_url": f"/download/prepared/{file_id}",
    })


@app.get("/download/prepared/{file_id}")
def download_prepared(file_id: str):
    path = PREPARED_DIR / f"{file_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/csv", filename="prepared_data.csv")


class TrainRequest(BaseModel):
    prepared_file_id: str
    columns: list[str]
    label: str
    label_kind: str  # "numeric" -> regression, "categorical" -> classification


_ID_COL_RE = re.compile(r"^id$|^id[_-]|[_-]id$", re.IGNORECASE)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


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


@app.post("/train/init")
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


@app.get("/train/stream/{job_id}")
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


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    model_filename: str = Form(...),
    feature_columns: str = Form(...),
    label: str = Form(...),
    task_type: str = Form(...),
    encoding_info: str = Form(...),
    scaling_info: str = Form(...),
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
        "columns": out_cols,
        "preview": preview,
        "download_url": f"/download/predictions/{pred_id}",
        "submission_url": f"/download/submission/{pred_id}",
        "submission_id_col": id_col or "id (row index)",
    })


@app.get("/download/predictions/{pred_id}")
def download_predictions(pred_id: str):
    path = PREDICTIONS_DIR / f"{pred_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/csv", filename="predictions.csv")


@app.get("/download/submission/{pred_id}")
def download_submission(pred_id: str):
    path = PREDICTIONS_DIR / f"{pred_id}_submission.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/csv", filename="submission.csv")


@app.get("/health")
def health():
    return {"status": "ok", "gpu": GPU_AVAILABLE}


@app.get("/resources")
def resources():
    return {
        "cpu_pct": _read_cpu(),
        "ram": _read_ram(),
        "gpu": _read_gpu(),
        "gpu_available": GPU_AVAILABLE,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
