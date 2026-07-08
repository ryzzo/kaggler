"""Feature/label analysis endpoint (/analyze) — stats, histograms, correlations."""
import csv
import io
import math
from collections import Counter
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from common import UPLOAD_DIR, _analysis_cache, _to_float

router = APIRouter()


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


@router.post("/analyze")
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

    all_cols_list = [*req.features, req.label]
    _analysis_cache[req.file_id] = {
        "numeric_cols":     [c for c in all_cols_list if c in numeric_cols],
        "categorical_cols": [c for c in all_cols_list if c not in numeric_cols],
        "missing_pct": {
            col: round(_missing_count(raw[col]) / len(rows) * 100, 2)
            for col in all_cols_list
        },
        "missingness_correlation": missingness_correlation,
        "row_count": len(rows),
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
