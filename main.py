import io
import csv
import math
import re
import uuid
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

PREPARED_DIR = Path("prepared")
PREPARED_DIR.mkdir(exist_ok=True)

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

    return JSONResponse({
        "file_id": req.file_id,
        "row_count": len(rows),
        "features": req.features,
        "label": label_info,
        "feature_analysis": feature_analysis,
        "correlation_matrix": {"columns": num_keys, "values": corr_matrix},
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


@app.post("/train")
def train(req: TrainRequest):
    path = PREPARED_DIR / f"{req.prepared_file_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Prepared file not found")
    if req.label not in req.columns:
        raise HTTPException(status_code=400, detail="Label is not in columns")
    excluded_id_cols = [c for c in req.columns if c != req.label and _ID_COL_RE.search(c.strip())]
    feature_cols = [c for c in req.columns if c != req.label and c not in excluded_id_cols]
    if not feature_cols:
        raise HTTPException(status_code=400, detail="No feature columns to train on")
    if req.label_kind not in ("numeric", "categorical"):
        raise HTTPException(status_code=400, detail="label_kind must be 'numeric' or 'categorical'")

    text = path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if len(rows) < 5:
        raise HTTPException(status_code=400, detail="Not enough rows to train/test split")

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

    if is_classification:
        models = {
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "lightgbm": LGBMClassifier(random_state=42, verbose=-1),
            "catboost": CatBoostClassifier(random_state=42, verbose=False),
        }
    else:
        models = {
            "gradient_boosting": GradientBoostingRegressor(random_state=42),
            "lightgbm": LGBMRegressor(random_state=42, verbose=-1),
            "catboost": CatBoostRegressor(random_state=42, verbose=False),
        }

    results = {}
    test_preds: dict[str, np.ndarray] = {}
    test_probas: dict[str, np.ndarray] = {}
    classes_: np.ndarray | None = None
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = np.asarray(model.predict(X_test)).ravel()
            test_preds[name] = preds
            if is_classification:
                results[name] = {
                    "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                    "f1_weighted": round(float(f1_score(y_test, preds, average="weighted")), 4),
                }
                proba = model.predict_proba(X_test)
                if classes_ is None:
                    classes_ = np.asarray(model.classes_)
                if np.array_equal(np.asarray(model.classes_), classes_):
                    test_probas[name] = np.asarray(proba)
            else:
                results[name] = {
                    "rmse": round(float(math.sqrt(mean_squared_error(y_test, preds))), 4),
                    "mae": round(float(mean_absolute_error(y_test, preds)), 4),
                    "r2": round(float(r2_score(y_test, preds)), 4),
                }
        except Exception as exc:
            results[name] = {"error": str(exc)}

    # ── Voting ensembles over every combination of 2+ successfully trained models ──
    ok_names = [n for n in models if n in test_preds]
    ensemble_results = []
    for r in (2, 3):
        for combo in combinations(ok_names, r):
            if r > len(ok_names):
                continue
            if is_classification:
                # soft voting: average predicted class probabilities across member models
                if all(n in test_probas for n in combo):
                    avg_proba = np.mean([test_probas[n] for n in combo], axis=0)
                    soft_pred = classes_[np.argmax(avg_proba, axis=1)]
                    ensemble_results.append({
                        "models": list(combo),
                        "voting": "soft",
                        "accuracy": round(float(accuracy_score(y_test, soft_pred)), 4),
                        "f1_weighted": round(float(f1_score(y_test, soft_pred, average="weighted")), 4),
                    })
            else:
                avg_pred = np.mean([test_preds[n] for n in combo], axis=0)
                ensemble_results.append({
                    "models": list(combo),
                    "voting": "average",
                    "rmse": round(float(math.sqrt(mean_squared_error(y_test, avg_pred))), 4),
                    "mae": round(float(mean_absolute_error(y_test, avg_pred)), 4),
                    "r2": round(float(r2_score(y_test, avg_pred)), 4),
                })

    return JSONResponse({
        "task_type": "classification" if is_classification else "regression",
        "label": req.label,
        "feature_columns": feature_cols,
        "excluded_id_columns": excluded_id_cols,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "results": results,
        "ensemble_results": ensemble_results,
    })


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
