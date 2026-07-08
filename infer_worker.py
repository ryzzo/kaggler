"""
Runs inside .venv-automl (Python 3.11 + pycaret) as a subprocess launched by
main.py. Never imported by the main app — invoked via
`python infer_worker.py <args-json-file>`.

Runs real inference for a model selected on the Hyperparameter Search page.
Reconstructs the exact PyCaret preprocessing pipeline used during training
(same session_id/params as tune_worker.py's full-dataset stage, so it's
deterministic) on the full clean training data, then either reuses an
already-fitted model (if a hyperparameter search saved one — the fast path)
or fits fresh with the given hyperparameters (the "baseline"/untuned path).
Predictions run through that same reconstructed pipeline via PyCaret's
predict_model(), so encoding, scaling, missing values, and unseen test-time
categories are all handled exactly as PyCaret would during training — no
separate manual encoding/imputation step needed for this path.

Emits one JSON object per line to stdout, forwarded live as SSE events by
routers/inference.py.
"""
import json
import re
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

from tune_worker import _load_model_classes, build_model, fit_with_progress

_ID_COL_RE = re.compile(r"^id$|^id[_-]|[_-]id$", re.IGNORECASE)


def emit(event: str, **data) -> None:
    print(json.dumps({"event": event, **data}), flush=True)


def main() -> None:
    args = json.loads(Path(sys.argv[1]).read_text())
    clean_csv_path = args["clean_csv_path"]
    test_csv_path = args["test_csv_path"]
    feature_cols: list[str] = args["feature_cols"]
    label: str = args["label"]
    label_kind: str = args["label_kind"]
    model_id: str = args["model_id"]
    best_params: dict = args.get("best_params") or {}
    model_path = args.get("model_path")  # optional saved joblib from a hyperparameter search run
    predictions_dir = Path(args["predictions_dir"])
    pred_id: str = args["pred_id"]

    if label_kind not in ("categorical", "numeric"):
        emit("error", detail="label_kind must be 'categorical' or 'numeric'")
        return

    is_classification = label_kind == "categorical"
    if is_classification:
        from pycaret.classification import ClassificationExperiment as Experiment
    else:
        from pycaret.regression import RegressionExperiment as Experiment

    emit("setup_start")
    df = pd.read_csv(clean_csv_path)
    df = df[feature_cols + [label]]

    exp = Experiment()
    exp.setup(data=df, target=label, session_id=42, train_size=0.8,
              n_jobs=1, verbose=False, html=False)
    emit("setup_done", n_train=len(exp.X_train), n_test=len(exp.X_test))

    if model_path and Path(model_path).exists():
        emit("loading_saved_model", model=model_id)
        bundle = joblib.load(model_path)
        model = bundle["model"]
    else:
        emit("fitting_start", model=model_id)
        _load_model_classes()
        t0 = time.perf_counter()
        model = build_model(label_kind, model_id, best_params)
        fit_with_progress(model, exp.X_train_transformed, exp.y_train_transformed, model_id, emit)
        emit("fitting_done", elapsed_s=round(time.perf_counter() - t0, 2))

    orig_test_df = pd.read_csv(test_csv_path)
    missing_cols = set(feature_cols) - set(orig_test_df.columns)
    if missing_cols:
        emit("error", detail=f"Test file is missing required columns: {sorted(missing_cols)}")
        return

    emit("predicting_start", n_rows=len(orig_test_df))
    test_features = orig_test_df[feature_cols]
    result = exp.predict_model(model, data=test_features, verbose=False)
    predictions = result["prediction_label"].tolist()

    fieldnames = orig_test_df.columns.tolist()
    pred_col = f"{label}_prediction"
    out_df = orig_test_df.copy()
    out_df[pred_col] = predictions

    predictions_dir.mkdir(exist_ok=True)
    out_path = predictions_dir / f"{pred_id}.csv"
    out_df.to_csv(out_path, index=False)

    id_col = next((c for c in fieldnames if _ID_COL_RE.match(c)), None)
    sub_path = predictions_dir / f"{pred_id}_submission.csv"
    sub_df = pd.DataFrame({
        "id" if id_col is None else id_col: (orig_test_df[id_col] if id_col else range(len(orig_test_df))),
        label: predictions,
    })
    sub_df.to_csv(sub_path, index=False)

    # Round-trip through pandas' own JSON encoder so numpy scalar types (int64,
    # float64, etc.) come out as plain JSON-safe values.
    preview = json.loads(out_df.head(10).to_json(orient="records"))

    emit("done",
         row_count=len(out_df),
         model_used=model_id,
         prediction_column=pred_col,
         columns=fieldnames + [pred_col],
         preview=preview,
         pred_id=pred_id,
         submission_id_col=id_col or "id (row index)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit("error", detail=str(exc))
        raise
