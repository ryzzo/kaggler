"""
Runs inside .venv-automl (Python 3.11 + pycaret) as a subprocess launched by main.py.
Never imported by the main app — invoked via `python automl_worker.py <args-json-file>`.

Emits one JSON object per line to stdout as progress happens, so the parent process
can forward each line as an SSE event in real time. All heavy pycaret/sklearn state
stays in this process; the parent only ever sees JSON.
"""
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

# Turbo-safe model set: excludes models known to be extremely slow or degenerate
# on this kind of tabular data (rbfsvm, gpc, mlp) — same set pycaret's compare_models
# turbo=True would use.
CANDIDATE_MODELS = [
    "lr", "knn", "nb", "dt", "svm", "ridge", "rf", "qda",
    "ada", "gbc", "lda", "et", "lightgbm", "catboost", "dummy",
]

# Models without predict_proba — shown on the leaderboard for comparison, but
# never eligible for the top-3 selection since soft voting requires probabilities.
NO_PROBA_MODELS = {"svm", "ridge"}


def emit(event: str, **data) -> None:
    print(json.dumps({"event": event, **data}), flush=True)


def _mean_row(df) -> dict:
    """Pull the CV 'Mean' row after create_model, or the single holdout row after predict_model."""
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


def _classification_metrics(y_true, y_pred, proba, classes) -> dict:
    """Same metric set pycaret reports, computed directly — used for soft-vote
    ensembles since we already have fitted models and just need holdout scores,
    not another CV refit (that's what pycaret's blend_models does by default)."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "Prec.": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    try:
        if len(classes) == 2:
            auc = roc_auc_score(y_true, proba[:, 1])
        else:
            auc = roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")
    except ValueError:
        auc = 0.0
    metrics["AUC"] = auc
    return {k: round(float(v), 4) for k, v in metrics.items()}


def main() -> None:
    args = json.loads(Path(sys.argv[1]).read_text())
    csv_path = args["csv_path"]
    feature_cols: list[str] = args["feature_cols"]
    label: str = args["label"]
    label_kind: str = args["label_kind"]  # "categorical" -> classification only, for now
    sample_frac: float = args.get("sample_frac", 0.1)
    models_dir = Path(args["models_dir"])
    run_id: str = args["run_id"]

    if label_kind != "categorical":
        emit("error", detail="AutoML currently supports classification labels only")
        return

    from pycaret.classification import ClassificationExperiment

    df = pd.read_csv(csv_path)
    df = df[feature_cols + [label]]

    n_total = len(df)
    sample = df.groupby(label, group_keys=False).apply(
        lambda g: g.sample(frac=sample_frac, random_state=42)
    )
    emit("sample_ready", n_total=n_total, n_sample=len(sample), sample_frac=sample_frac)

    # ── Stage 1: compare candidate models on the 10% sample ──
    exp = ClassificationExperiment()
    exp.setup(data=sample, target=label, session_id=42, train_size=0.8,
              n_jobs=1, verbose=False, html=False)

    available = set(exp.models().index.tolist())
    model_ids = [m for m in CANDIDATE_MODELS if m in available]
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

    ranked = sorted(leaderboard, key=lambda r: r["metrics"].get("Accuracy", 0), reverse=True)
    eligible = [r for r in ranked if r["model"] not in NO_PROBA_MODELS]
    top3 = [r["model"] for r in eligible[:3]]
    emit("leaderboard_done", top3=top3)

    # ── Stage 2: retrain top 3 on the full dataset ──
    exp2 = ClassificationExperiment()
    exp2.setup(data=df, target=label, session_id=42, train_size=0.8,
               n_jobs=1, verbose=False, html=False)
    emit("full_setup", n_train=len(exp2.X_train), n_test=len(exp2.X_test))

    fitted = {}
    for idx, mid in enumerate(top3):
        t0 = time.perf_counter()
        try:
            model = exp2.create_model(mid, fold=5, verbose=False)
            fitted[mid] = model
            exp2.predict_model(model, verbose=False)
            metrics = _mean_row(exp2.pull())
            elapsed = round(time.perf_counter() - t0, 2)
            emit("individual_result", index=idx, model=mid, metrics=metrics, elapsed_s=elapsed)
        except Exception as exc:
            emit("individual_result", index=idx, model=mid, error=str(exc),
                 elapsed_s=round(time.perf_counter() - t0, 2))

    # ── Stage 3: soft-voting ensembles over all 2- and 3- combinations of top 3 ──
    # Averages predict_proba from the already-fitted models on the held-out test
    # split directly (pycaret's blend_models() instead refits a fresh VotingClassifier
    # via k-fold CV — for 3 tree models on the full dataset that's the individual
    # retrain cost all over again, times the fold count, times every combination).
    ok_ids = [m for m in top3 if m in fitted]
    X_test = exp2.X_test_transformed
    y_test = exp2.y_test_transformed
    for r in (2, 3):
        for combo in combinations(ok_ids, r):
            t0 = time.perf_counter()
            try:
                probas = [fitted[m].predict_proba(X_test) for m in combo]
                avg_proba = np.mean(probas, axis=0)
                classes = fitted[combo[0]].classes_
                pred = classes[np.argmax(avg_proba, axis=1)]
                metrics = _classification_metrics(y_test, pred, avg_proba, classes)
                elapsed = round(time.perf_counter() - t0, 2)
                emit("ensemble_result", models=list(combo), voting="soft",
                     metrics=metrics, elapsed_s=elapsed)
            except Exception as exc:
                emit("ensemble_result", models=list(combo), voting="soft",
                     error=str(exc), elapsed_s=round(time.perf_counter() - t0, 2))

    # ── Save top-3 individual models (pycaret-native pickle; loadable only from this venv) ──
    models_dir.mkdir(exist_ok=True)
    saved = []
    for mid in ok_ids:
        filename = f"{run_id}__automl_{mid}"
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
