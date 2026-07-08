"""
Runs inside .venv-automl (Python 3.11 + pycaret) as a subprocess launched by main.py.
Never imported by the main app — invoked via `python automl_worker.py <args-json-file>`.

Data Training's job is model *scouting* only: compare a broad set of candidate
models on a 10% stratified sample and report how each performs, so the user can
pick which ones look worth pursuing. Actual training (and hyperparameter search)
happens separately on the Hyperparameter Search page via tune_worker.py.

Supports both classification (categorical label) and regression (numeric label).

Emits one JSON object per line to stdout as progress happens, so the parent
process can forward each line as an SSE event in real time.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Turbo-safe model sets: exclude models known to be extremely slow or degenerate
# on this kind of tabular data — same idea as pycaret's compare_models turbo=True.
CANDIDATE_MODELS_CLASSIFICATION = [
    "lr", "knn", "nb", "dt", "svm", "ridge", "rf", "qda",
    "ada", "gbc", "lda", "et", "lightgbm", "catboost", "dummy",
]
CANDIDATE_MODELS_REGRESSION = [
    "lr", "lasso", "ridge", "en", "lar", "llar", "omp", "br", "par",
    "huber", "svm", "knn", "dt", "rf", "et", "ada", "gbr", "lightgbm", "catboost", "dummy",
]

# Ranking cascade per task: primary metric first, then tie-breakers in order —
# a model only falls back to the next metric when every metric before it is tied.
RANK_METRICS = {
    "categorical": ["Accuracy", "F1", "MCC", "Kappa", "AUC"],
    "numeric": ["R2", "RMSE", "MAE", "MSE"],
}
# Metrics where a *lower* value is better (everything else is higher-is-better).
LOWER_IS_BETTER = {"RMSE", "MAE", "MSE", "RMSLE", "MAPE"}


def _rank_tuple(metrics: dict, label_kind: str) -> tuple:
    """Sort key for ranking models: compares each metric in RANK_METRICS in turn,
    only moving to the next one when the previous is tied (equal after rounding).
    Values are sign-flipped for lower-is-better metrics so a plain descending sort
    on the tuple always means 'better'."""
    out = []
    for key in RANK_METRICS[label_kind]:
        v = metrics.get(key, 0.0)
        out.append(-v if key in LOWER_IS_BETTER else v)
    return tuple(out)


def emit(event: str, **data) -> None:
    print(json.dumps({"event": event, **data}), flush=True)


def _mean_row(df) -> dict:
    """Pull the CV 'Mean' row after create_model."""
    row = df.loc["Mean"] if "Mean" in df.index else df.iloc[0]
    out = {}
    for k, v in row.items():
        if k == "Model":
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def main() -> None:
    args = json.loads(Path(sys.argv[1]).read_text())
    csv_path = args["csv_path"]
    feature_cols: list[str] = args["feature_cols"]
    label: str = args["label"]
    label_kind: str = args["label_kind"]  # "categorical" -> classification, "numeric" -> regression
    sample_frac: float = args.get("sample_frac", 0.1)

    if label_kind not in ("categorical", "numeric"):
        emit("error", detail="label_kind must be 'categorical' or 'numeric'")
        return

    df = pd.read_csv(csv_path)
    df = df[feature_cols + [label]]
    n_total = len(df)

    if label_kind == "categorical":
        sample = df.groupby(label, group_keys=False).apply(
            lambda g: g.sample(frac=sample_frac, random_state=42)
        )
    else:
        sample = df.sample(frac=sample_frac, random_state=42)
    emit("sample_ready", n_total=n_total, n_sample=len(sample), sample_frac=sample_frac)

    # ── Compare candidate models on the sample ──
    if label_kind == "categorical":
        from pycaret.classification import ClassificationExperiment
        exp = ClassificationExperiment()
        candidate_models = CANDIDATE_MODELS_CLASSIFICATION
    else:
        from pycaret.regression import RegressionExperiment
        exp = RegressionExperiment()
        candidate_models = CANDIDATE_MODELS_REGRESSION

    exp.setup(data=sample, target=label, session_id=42, train_size=0.8,
              n_jobs=1, verbose=False, html=False)

    available = set(exp.models().index.tolist())
    model_ids = [m for m in candidate_models if m in available]
    emit("compare_start", models=model_ids, total=len(model_ids))

    leaderboard = []
    for idx, mid in enumerate(model_ids):
        t0 = time.perf_counter()
        try:
            exp.create_model(mid, fold=5, verbose=False)
            metrics = _mean_row(exp.pull())
            elapsed = round(time.perf_counter() - t0, 2)
            row = {"model": mid, "metrics": metrics, "elapsed_s": elapsed}
            leaderboard.append(row)
            emit("compare_result", index=idx, **row)
        except Exception as exc:
            emit("compare_result", index=idx, model=mid, error=str(exc),
                 elapsed_s=round(time.perf_counter() - t0, 2))

    ranked = sorted(leaderboard, key=lambda r: _rank_tuple(r["metrics"], label_kind), reverse=True)
    top3 = [r["model"] for r in ranked[:3]]
    emit("leaderboard_done", top3=top3)

    emit("done")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit("error", detail=str(exc))
        raise
