"""
EXPERIMENT (branch experiment/pytabkit): drop-in alternative to automl_worker.py that
scouts models with pytabkit instead of PyCaret. Same args-json, same JSON-lines event
protocol (sample_ready, gpu_info, compare_start, compare_result, leaderboard_done,
best_model_fit_start, best_model_saved, done, error), so routers/training.py and the
front end need no changes.

Runs in .venv-pytabkit (see requirements-pytabkit.txt), selected with
KAGGLER_SCOUT_ENGINE=pytabkit.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn import metrics as skm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from gpu import cuda_available, describe as gpu_describe
from tabular_prep import FittedBundleModel, TabularPrep

# pytabkit class-name stems; "_Classifier"/"_Regressor" is appended. Missing ones are skipped.
CANDIDATES = ["RealMLP_TD", "TabM_D", "LGBM_TD", "XGB_TD", "CatBoost_TD"]

RANK_METRICS = {
    "categorical": ["Accuracy", "F1", "MCC", "Kappa", "AUC"],
    "numeric": ["R2", "RMSE", "MAE", "MSE"],
}
LOWER_IS_BETTER = {"RMSE", "MAE", "MSE"}


def _rank_tuple(m: dict, kind: str) -> tuple:
    return tuple(-m.get(k, 0.0) if k in LOWER_IS_BETTER else m.get(k, 0.0) for k in RANK_METRICS[kind])


def emit(event: str, **data) -> None:
    print(json.dumps({"event": event, **data}), flush=True)


def _fold_metrics(kind, y_true, y_pred, proba, n_classes) -> dict:
    if kind == "numeric":
        mse = skm.mean_squared_error(y_true, y_pred)
        return {"R2": skm.r2_score(y_true, y_pred), "RMSE": float(np.sqrt(mse)),
                "MAE": skm.mean_absolute_error(y_true, y_pred), "MSE": mse}
    out = {"Accuracy": skm.accuracy_score(y_true, y_pred),
           "F1": skm.f1_score(y_true, y_pred, average="weighted"),
           "MCC": skm.matthews_corrcoef(y_true, y_pred),
           "Kappa": skm.cohen_kappa_score(y_true, y_pred)}
    try:
        out["AUC"] = (skm.roc_auc_score(y_true, proba[:, 1]) if n_classes == 2 else
                      skm.roc_auc_score(y_true, proba, multi_class="ovr", average="weighted",
                                        labels=list(range(n_classes))))
    except Exception:
        out["AUC"] = 0.0
    return out


def _make(name: str, kind: str, device: str):
    import pytabkit
    cls = getattr(pytabkit, f"{name}_{'Classifier' if kind == 'categorical' else 'Regressor'}", None)
    if cls is None:
        return None
    try:
        return cls(device=device, random_state=42, verbosity=0)
    except TypeError:
        return cls(device=device, random_state=42)


def main() -> None:
    args = json.loads(Path(sys.argv[1]).read_text())
    feature_cols, label, kind = args["feature_cols"], args["label"], args["label_kind"]
    sample_frac = args.get("sample_frac", 0.1)
    models_dir, run_id = Path(args["models_dir"]), args["run_id"]
    if kind not in ("categorical", "numeric"):
        emit("error", detail="label_kind must be 'categorical' or 'numeric'")
        return

    df = pd.read_csv(args["csv_path"])[feature_cols + [label]].dropna(subset=[label])
    if kind == "categorical":
        sample = df.groupby(label, group_keys=False).apply(lambda g: g.sample(frac=sample_frac, random_state=42))
    else:
        sample = df.sample(frac=sample_frac, random_state=42)
    emit("sample_ready", n_total=len(df), n_sample=len(sample), sample_frac=sample_frac)

    device = "cuda" if cuda_available() else "cpu"
    emit("gpu_info", **gpu_describe(), engine="pytabkit", device=device)

    le = None
    def target(frame):
        nonlocal le
        if kind == "categorical":
            if le is None:
                le = LabelEncoder().fit(df[label].astype(str))
            return le.transform(frame[label].astype(str))
        return frame[label].to_numpy(dtype=float)

    X, y = sample[feature_cols].reset_index(drop=True), target(sample)
    n_classes = len(le.classes_) if le is not None else 0
    folds = (StratifiedKFold(5, shuffle=True, random_state=42) if kind == "categorical"
             else KFold(5, shuffle=True, random_state=42))

    names = [n for n in CANDIDATES if _make(n, kind, device) is not None]
    emit("compare_start", models=names, total=len(names))

    board = []
    for idx, name in enumerate(names):
        t0 = time.perf_counter()
        try:
            per_fold = []
            for tr, te in folds.split(X, y):
                prep = TabularPrep()
                est = _make(name, kind, device)
                est.fit(prep.fit_transform(X.iloc[tr]), y[tr])
                Xte = prep.transform(X.iloc[te])
                proba = est.predict_proba(Xte) if kind == "categorical" else None
                per_fold.append(_fold_metrics(kind, y[te], est.predict(Xte), proba, n_classes))
            mean = {k: float(np.mean([f[k] for f in per_fold])) for k in per_fold[0]}
            row = {"model": name, "metrics": mean, "elapsed_s": round(time.perf_counter() - t0, 2)}
            board.append(row)
            emit("compare_result", index=idx, **row)
        except Exception as exc:
            emit("compare_result", index=idx, model=name, error=str(exc),
                 elapsed_s=round(time.perf_counter() - t0, 2))

    ranked = sorted(board, key=lambda r: _rank_tuple(r["metrics"], kind), reverse=True)
    top3 = [r["model"] for r in ranked[:3]]
    emit("leaderboard_done", top3=top3)

    if top3:
        best = top3[0]
        emit("best_model_fit_start", model=best)
        t0 = time.perf_counter()
        try:
            Xf, yf = df[feature_cols].reset_index(drop=True), target(df)
            prep = TabularPrep().fit(Xf)
            est = _make(best, kind, device)
            est.fit(prep.transform(Xf), yf)
            models_dir.mkdir(exist_ok=True)
            filename = f"{run_id}__pytabkit_{best}.joblib"
            joblib.dump({"model": FittedBundleModel(prep, est, le.classes_ if le is not None else None),
                         "model_id": best, "engine": "pytabkit"}, models_dir / filename)
            emit("best_model_saved", model=best, filename=filename, elapsed_s=round(time.perf_counter() - t0, 2))
        except Exception as exc:
            emit("best_model_saved", model=best, error=str(exc), elapsed_s=round(time.perf_counter() - t0, 2))
    emit("done")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit("error", detail=str(exc))
        raise
