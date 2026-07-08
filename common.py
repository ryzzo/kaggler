"""Shared state and helpers used across routers: storage locations, the
in-memory analysis cache, and small utilities with no natural single owner."""
import json
import re
from pathlib import Path
from typing import Any

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

PREPARED_DIR = Path("prepared")
PREPARED_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

PREDICTIONS_DIR = Path("predictions")
PREDICTIONS_DIR.mkdir(exist_ok=True)

# PyCaret needs a much older numpy/pandas/scikit-learn stack than the rest of this
# app, and has no wheels for the Python version this app runs on — so it lives in
# its own venv and is only ever invoked as a subprocess (see automl_worker.py).
AUTOML_VENV_PYTHON = Path(".venv-automl/bin/python")
AUTOML_WORKER_SCRIPT = Path("automl_worker.py")
TUNE_WORKER_SCRIPT = Path("tune_worker.py")

# In-memory analysis cache: populated by /analyze, consumed by /prepare.
# Keys are file_ids; automatically cleared on server restart.
_analysis_cache: dict[str, dict] = {}

_ID_COL_RE = re.compile(r"^id$|^id[_-]|[_-]id$", re.IGNORECASE)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _to_float(values: list[str]) -> list[float] | None:
    try:
        return [float(v) for v in values if v.strip() != ""]
    except ValueError:
        return None
