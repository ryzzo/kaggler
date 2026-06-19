import io
import csv
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="File Processing API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


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


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
