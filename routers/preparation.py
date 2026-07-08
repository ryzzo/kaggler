"""Missing-value imputation + encode/scale endpoint (/prepare) and its downloads."""
import csv
import io
import math
import uuid
from collections import Counter
from typing import Any

import joblib
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from common import PREPARED_DIR, UPLOAD_DIR, _analysis_cache, _to_float

router = APIRouter()


def _label_encode(values: list[str]) -> tuple[list[int], dict[str, int]]:
    uniques = sorted(set(values))
    mapping = {v: i for i, v in enumerate(uniques)}
    return [mapping[v] for v in values], mapping


def _impute_missing(
    rows: list[dict[str, str]],
    cols: list[str],
    column_kinds: dict[str, str],
    file_id: str,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Impute missing values in `rows` (mutates in-place) before encoding/scaling.
    Returns (imputation_info, imputation_params):
      - imputation_info: human-readable metadata for the UI
      - imputation_params: serialisable bundle applied identically at inference
        Each entry: {"method": ..., "value"/"model"/"feature_cols"/"col_means": ...}
    """
    cache = _analysis_cache.get(file_id, {})
    miss_corr_cache: dict[str, dict] = cache.get("missingness_correlation", {})
    n = len(rows)
    imputation_info: dict[str, dict] = {}
    imputation_params: dict[str, dict] = {}

    def _is_blank(v: str) -> bool:
        return v.strip() == ""

    def _missing_count_col(col: str) -> int:
        return sum(1 for r in rows if _is_blank(r[col]))

    def _col_mean(col: str) -> float:
        vals = [float(r[col]) for r in rows if not _is_blank(r[col])]
        return sum(vals) / len(vals) if vals else 0.0

    def _knn_feature_cols(exclude: str) -> list[str]:
        return [
            c for c in cols
            if c != exclude
            and column_kinds.get(c, "numeric") == "numeric"
            and _missing_count_col(c) / n < 0.05
        ]

    def _max_abs_r(col: str) -> float:
        top = miss_corr_cache.get(col, {}).get("top_correlations", [])
        return max((abs(e["r"]) for e in top), default=0.0)

    def _apply_median(col: str, fill: float) -> None:
        for r in rows:
            if _is_blank(r[col]):
                r[col] = str(fill)

    def _apply_mode(col: str, fill: str) -> None:
        for r in rows:
            if _is_blank(r[col]):
                r[col] = fill

    def _knn_fill(col: str, kind: str, missing_count: int, missing_pct: float) -> dict:
        feat_cols = _knn_feature_cols(col)
        if len(feat_cols) < 1:
            if kind == "numeric":
                vals = sorted(float(r[col]) for r in rows if not _is_blank(r[col]))
                fill = vals[len(vals) // 2] if vals else 0.0
                _apply_median(col, fill)
                imputation_params[col] = {"method": "median_fallback", "value": fill}
                return {"method": "median_fallback", "value": round(fill, 6),
                        "missing_count": missing_count, "missing_pct": missing_pct,
                        "note": "KNN skipped — no usable numeric feature columns; used median instead"}
            else:
                cnt = Counter(r[col] for r in rows if not _is_blank(r[col]))
                fill = cnt.most_common(1)[0][0] if cnt else ""
                _apply_mode(col, fill)
                imputation_params[col] = {"method": "mode_fallback", "value": fill}
                return {"method": "mode_fallback", "value": fill,
                        "missing_count": missing_count, "missing_pct": missing_pct,
                        "note": "KNN skipped — no usable numeric feature columns; used mode instead"}

        col_means = {c: _col_mean(c) for c in feat_cols}
        donor_rows = [r for r in rows if not _is_blank(r[col])]

        if len(donor_rows) > 5000:
            donor_rows.sort(key=lambda r: float(r[feat_cols[0]]) if not _is_blank(r[feat_cols[0]]) else col_means[feat_cols[0]])
            step = len(donor_rows) / 5000
            donor_rows = [donor_rows[int(i * step)] for i in range(5000)]

        def row_to_x(r: dict[str, str]) -> list[float]:
            return [float(r[c]) if not _is_blank(r[c]) else col_means[c] for c in feat_cols]

        X_donor = [row_to_x(r) for r in donor_rows]

        if kind == "numeric":
            y_donor = [float(r[col]) for r in donor_rows]
            knn: Any = KNeighborsRegressor(n_neighbors=min(5, len(donor_rows)))
            knn.fit(X_donor, y_donor)
            for r in rows:
                if _is_blank(r[col]):
                    r[col] = str(round(float(knn.predict([row_to_x(r)])[0]), 6))
        else:
            y_donor = [r[col] for r in donor_rows]
            if len(set(y_donor)) < 2:
                cnt = Counter(y_donor)
                fill = cnt.most_common(1)[0][0]
                _apply_mode(col, fill)
                imputation_params[col] = {"method": "mode_fallback", "value": fill}
                return {"method": "mode_fallback", "value": fill,
                        "missing_count": missing_count, "missing_pct": missing_pct,
                        "note": "KNN skipped — only one class in donors; used mode instead"}
            knn = KNeighborsClassifier(n_neighbors=min(5, len(donor_rows)))
            knn.fit(X_donor, y_donor)
            for r in rows:
                if _is_blank(r[col]):
                    r[col] = str(knn.predict([row_to_x(r)])[0])

        imputation_params[col] = {
            "method": "knn",
            "model": knn,
            "feature_cols": feat_cols,
            "col_means": col_means,
        }
        return {
            "method": "knn",
            "k": min(5, len(donor_rows)),
            "n_donors": len(donor_rows),
            "knn_features": feat_cols,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
        }

    for col in cols:
        mc = _missing_count_col(col)
        if mc == 0:
            continue
        pct = round(mc / n * 100, 2)
        kind = column_kinds.get(col, "numeric")
        max_r = _max_abs_r(col)
        is_weak = max_r < 0.4

        if pct > 30:
            imputation_info[col] = {
                "method": "skipped", "missing_count": mc, "missing_pct": pct,
                "note": f">{pct}% missing — too high for automatic imputation; blanks encoded as __missing__",
            }
            imputation_params[col] = {"method": "skipped"}
        elif pct < 5:
            if kind == "numeric" and is_weak:
                vals = sorted(float(r[col]) for r in rows if not _is_blank(r[col]))
                fill = vals[len(vals) // 2] if vals else 0.0
                _apply_median(col, fill)
                imputation_info[col] = {
                    "method": "median", "value": round(fill, 6),
                    "missing_count": mc, "missing_pct": pct,
                    "max_missingness_r": round(max_r, 4),
                }
                imputation_params[col] = {"method": "median", "value": fill}
            elif kind == "numeric" and not is_weak:
                imputation_info[col] = _knn_fill(col, kind, mc, pct)
                imputation_info[col]["note"] = f"Missingness correlated (max |r|={round(max_r,4)}) — used KNN instead of median"
            else:
                cnt = Counter(r[col] for r in rows if not _is_blank(r[col]))
                fill = cnt.most_common(1)[0][0] if cnt else ""
                _apply_mode(col, fill)
                imputation_info[col] = {
                    "method": "mode", "value": fill,
                    "missing_count": mc, "missing_pct": pct,
                }
                imputation_params[col] = {"method": "mode", "value": fill}
        else:
            imputation_info[col] = _knn_fill(col, kind, mc, pct)

    return imputation_info, imputation_params


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


@router.post("/prepare")
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

    # Resolve column kinds before imputation (same logic as below)
    column_kinds: dict[str, str] = {}
    for col in cols:
        raw_values = [r[col] for r in rows]
        column_kinds[col] = req.column_kinds.get(col) or (
            "numeric" if _to_float(raw_values) is not None else "categorical"
        )

    # Impute missing values in-place before encoding/scaling
    imputation_info, imputation_params = _impute_missing(rows, cols, column_kinds, req.file_id)

    # Persist imputation params bundle (fitted models + fill values) for inference
    imp_id = str(uuid.uuid4())
    joblib.dump(imputation_params, PREPARED_DIR / f"{imp_id}__imputation.joblib")

    # Save the clean (imputed, untransformed) CSV for inspection and training reuse
    clean_id = str(uuid.uuid4())
    clean_path = PREPARED_DIR / f"{clean_id}__clean.csv"
    with clean_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for r in rows:
            writer.writerow([r[col] for col in cols])
    # Cache the clean file path so training can reference it
    if req.file_id in _analysis_cache:
        _analysis_cache[req.file_id]["clean_file_id"] = clean_id

    prepared: dict[str, list[float]] = {}
    encoding_info: dict[str, dict[str, int]] = {}
    scaling_info: dict[str, dict[str, float]] = {}

    for col in cols:
        raw_values = [r[col] for r in rows]
        kind = column_kinds[col]

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
        "clean_file_id": clean_id,
        "imputation_params_id": imp_id,
        "row_count": len(rows),
        "columns": cols,
        "encoding_info": encoding_info,
        "scaling_info": scaling_info,
        "imputation_info": imputation_info,
        "preview": preview,
        "download_url": f"/download/prepared/{file_id}",
        "clean_download_url": f"/download/clean/{clean_id}",
    })


@router.get("/download/prepared/{file_id}")
def download_prepared(file_id: str):
    path = PREPARED_DIR / f"{file_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/csv", filename="prepared_data.csv")


@router.get("/download/clean/{file_id}")
def download_clean(file_id: str):
    path = PREPARED_DIR / f"{file_id}__clean.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/csv", filename="clean_data.csv")
