"""
Runs inside .venv-automl (Python 3.11 + pycaret) as a subprocess launched by main.py.
Never imported by the main app — invoked via `python tune_worker.py <args-json-file>`.

Hyperparameter search for one or more models already identified by Data Training
(automl_worker.py's top-3). For each selected model: baseline fit on the 10% sample,
then a randomized hyperparameter search on that same sample, then one final fit of
the best-found hyperparameters on the full dataset — same cheap-search/expensive-final-fit
shape as automl_worker.py, since a search that refit hundreds of times on the full
690k-row dataset would take hours.

Emits one JSON object per line to stdout, forwarded live as SSE events by
routers/hyperparameter.py.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd


def emit(event: str, **data) -> None:
    print(json.dumps({"event": event, **data}), flush=True)


def _mean_row(df) -> dict:
    """Pull the CV 'Mean' row after create_model/tune_model, or the single holdout row after predict_model."""
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


def _json_safe_params(params: dict) -> dict:
    """Model.get_params() can contain non-JSON-safe values (e.g. empty dict sentinels,
    numpy scalars) — coerce to plain str/float/int/bool/None for the JSON event."""
    out = {}
    for k, v in params.items():
        if v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def main() -> None:
    args = json.loads(Path(sys.argv[1]).read_text())
    csv_path = args["csv_path"]
    feature_cols: list[str] = args["feature_cols"]
    label: str = args["label"]
    label_kind: str = args["label_kind"]
    sample_frac: float = args.get("sample_frac", 0.1)
    model_ids: list[str] = args["model_ids"]
    n_iter: int = args.get("n_iter", 10)
    fold: int = args.get("fold", 5)
    models_dir = Path(args["models_dir"])
    run_id: str = args["run_id"]

    if label_kind != "categorical":
        emit("error", detail="Hyperparameter search currently supports classification labels only")
        return
    if not model_ids:
        emit("error", detail="No models selected to tune")
        return

    from pycaret.classification import ClassificationExperiment

    df = pd.read_csv(csv_path)
    df = df[feature_cols + [label]]

    n_total = len(df)
    sample = df.groupby(label, group_keys=False).apply(
        lambda g: g.sample(frac=sample_frac, random_state=42)
    )
    emit("sample_ready", n_total=n_total, n_sample=len(sample), sample_frac=sample_frac)

    # ── Stage 1: baseline + search on the 10% sample ──
    exp = ClassificationExperiment()
    exp.setup(data=sample, target=label, session_id=42, train_size=0.8,
              n_jobs=1, verbose=False, html=False)

    tuned_models: dict[str, object] = {}
    for idx, mid in enumerate(model_ids):
        t0 = time.perf_counter()
        try:
            baseline = exp.create_model(mid, fold=fold, verbose=False)
            baseline_metrics = _mean_row(exp.pull())
            elapsed = round(time.perf_counter() - t0, 2)
            emit("baseline_result", index=idx, model=mid, metrics=baseline_metrics, elapsed_s=elapsed)
        except Exception as exc:
            emit("baseline_result", index=idx, model=mid, error=str(exc),
                 elapsed_s=round(time.perf_counter() - t0, 2))
            continue

        t0 = time.perf_counter()
        try:
            tuned = exp.tune_model(baseline, fold=fold, n_iter=n_iter, verbose=False, choose_better=True)
            tuned_metrics = _mean_row(exp.pull())
            elapsed = round(time.perf_counter() - t0, 2)
            tuned_models[mid] = tuned
            emit("tune_result", index=idx, model=mid, metrics=tuned_metrics,
                 best_params=_json_safe_params(tuned.get_params()), elapsed_s=elapsed)
        except Exception as exc:
            emit("tune_result", index=idx, model=mid, error=str(exc),
                 elapsed_s=round(time.perf_counter() - t0, 2))

    if not tuned_models:
        emit("error", detail="Hyperparameter search failed for every selected model")
        return

    # ── Stage 2: refit the tuned hyperparameters once on the full dataset ──
    exp2 = ClassificationExperiment()
    exp2.setup(data=df, target=label, session_id=42, train_size=0.8,
               n_jobs=1, verbose=False, html=False)
    emit("full_setup", n_train=len(exp2.X_train), n_test=len(exp2.X_test))

    fitted = {}
    for idx, mid in enumerate(tuned_models):
        t0 = time.perf_counter()
        try:
            model = exp2.create_model(tuned_models[mid], fold=fold, verbose=False)
            fitted[mid] = model
            exp2.predict_model(model, verbose=False)
            metrics = _mean_row(exp2.pull())
            elapsed = round(time.perf_counter() - t0, 2)
            emit("full_refit_result", index=idx, model=mid, metrics=metrics, elapsed_s=elapsed)
        except Exception as exc:
            emit("full_refit_result", index=idx, model=mid, error=str(exc),
                 elapsed_s=round(time.perf_counter() - t0, 2))

    # ── Save tuned models (pycaret-native pickle; loadable only from this venv) ──
    models_dir.mkdir(exist_ok=True)
    saved = []
    for mid in fitted:
        filename = f"{run_id}__tuned_{mid}"
        exp2.save_model(fitted[mid], str(models_dir / filename), verbose=False)
        saved.append({"model": mid, "filename": f"{filename}.pkl"})
    emit("saved", saved_models=saved)

    emit("done")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit("error", detail=str(exc))
        raise
