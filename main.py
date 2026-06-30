import io
import csv
import math
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

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
    save_path = UPLOAD_DIR / f"{file_id}_prepared.csv"
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
    path = UPLOAD_DIR / f"{file_id}_prepared.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/csv", filename="prepared_data.csv")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
