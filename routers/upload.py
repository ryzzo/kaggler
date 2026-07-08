"""File upload + column selection endpoints (/upload, /upload/batch, /select-columns)."""
import csv
import io
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from common import UPLOAD_DIR

router = APIRouter()


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


@router.post("/upload")
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


@router.post("/upload/batch")
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


@router.post("/select-columns")
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
