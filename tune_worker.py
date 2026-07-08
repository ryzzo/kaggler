"""
Runs inside .venv-automl (Python 3.11 + pycaret + optuna) as a subprocess launched
by main.py. Never imported by the main app — invoked via
`python tune_worker.py <args-json-file>`.

Hyperparameter search for one or more models, with the search space (which
parameters, their ranges/choices) fully specified by the caller — this is what
lets the Hyperparameter Search page offer the user real control over what gets
tuned, rather than a fixed grid. Supports both classification (categorical
label) and regression (numeric label).

Uses PyCaret only for its setup() preprocessing pipeline (consistent encoding/
scaling with the rest of the app) — the actual search is raw Optuna driving
plain scikit-learn/LightGBM/CatBoost estimators, since that gives full control
over the search space and direct access to the Study object for
optuna.visualization plots and per-trial progress callbacks.

For each model: Optuna search on the 10% sample (cross-validated), then one
final fit of the best-found hyperparameters on the full dataset — same
cheap-search/expensive-final-fit shape as automl_worker.py.

Emits one JSON object per line to stdout, forwarded live as SSE events by
routers/hyperparameter.py.
"""
import json
import sys
import threading
import time
from pathlib import Path

import joblib
import pandas as pd

MODEL_CLASS_MAP: dict[str, dict] = {"categorical": {}, "numeric": {}}
FIXED_KWARGS: dict[str, dict] = {"categorical": {}, "numeric": {}}


def _load_model_classes():
    """Deferred import — keeps `python tune_worker.py --help`-style failures
    fast and isolates heavy imports (lightgbm/catboost) to when actually needed."""
    from sklearn.linear_model import (
        LogisticRegression, RidgeClassifier, LinearRegression, Ridge, Lasso,
        ElasticNet, HuberRegressor, PassiveAggressiveRegressor,
    )
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier,
        GradientBoostingClassifier, AdaBoostClassifier,
        RandomForestRegressor, ExtraTreesRegressor,
        GradientBoostingRegressor, AdaBoostRegressor,
    )
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    from catboost import CatBoostClassifier, CatBoostRegressor

    MODEL_CLASS_MAP["categorical"].update({
        "lr": LogisticRegression, "knn": KNeighborsClassifier, "nb": GaussianNB,
        "dt": DecisionTreeClassifier, "ridge": RidgeClassifier, "rf": RandomForestClassifier,
        "qda": QuadraticDiscriminantAnalysis, "ada": AdaBoostClassifier,
        "gbc": GradientBoostingClassifier, "lda": LinearDiscriminantAnalysis,
        "et": ExtraTreesClassifier, "lightgbm": LGBMClassifier,
        "catboost": CatBoostClassifier, "dummy": DummyClassifier,
    })
    FIXED_KWARGS["categorical"].update({
        "rf": {"random_state": 42}, "et": {"random_state": 42},
        "gbc": {"random_state": 42}, "dt": {"random_state": 42},
        "ada": {"random_state": 42}, "lr": {"random_state": 42, "max_iter": 1000},
        "ridge": {"random_state": 42}, "dummy": {"random_state": 42},
        "lightgbm": {"random_state": 42, "verbose": -1},
        "catboost": {"random_state": 42, "verbose": False},
        "knn": {}, "lda": {}, "nb": {}, "qda": {},
    })

    MODEL_CLASS_MAP["numeric"].update({
        "lr": LinearRegression, "ridge": Ridge, "lasso": Lasso, "en": ElasticNet,
        "huber": HuberRegressor, "par": PassiveAggressiveRegressor,
        "knn": KNeighborsRegressor, "dt": DecisionTreeRegressor, "svm": SVR,
        "rf": RandomForestRegressor, "et": ExtraTreesRegressor,
        "ada": AdaBoostRegressor, "gbr": GradientBoostingRegressor,
        "lightgbm": LGBMRegressor, "catboost": CatBoostRegressor, "dummy": DummyRegressor,
    })
    FIXED_KWARGS["numeric"].update({
        "rf": {"random_state": 42}, "et": {"random_state": 42},
        "gbr": {"random_state": 42}, "dt": {"random_state": 42},
        "ada": {"random_state": 42}, "ridge": {"random_state": 42},
        "lasso": {"random_state": 42}, "en": {"random_state": 42},
        "par": {"random_state": 42}, "dummy": {"random_state": 42},
        "lightgbm": {"random_state": 42, "verbose": -1},
        "catboost": {"random_state": 42, "verbose": False},
        "lr": {}, "huber": {}, "knn": {}, "svm": {},
    })


def emit(event: str, **data) -> None:
    print(json.dumps({"event": event, **data}), flush=True)


def build_model(label_kind: str, model_id: str, params: dict):
    cls = MODEL_CLASS_MAP[label_kind][model_id]
    kwargs = {**FIXED_KWARGS[label_kind].get(model_id, {}), **params}
    return cls(**kwargs)


def run_with_heartbeat(fn, model_id: str, emit_fn):
    """Runs the zero-arg callable `fn` in a background thread, emitting periodic
    elapsed-time 'fit_progress' heartbeats meanwhile, and returns fn()'s result.

    Used whenever there's no way to hook real per-step progress — an opaque call
    like pycaret's create_model(), or a model type fit_with_progress doesn't have
    a native callback for. No completion percentage, but proof the process is
    still alive during a long fit instead of a silent multi-minute wait."""
    t0 = time.perf_counter()
    result: dict = {}
    exc_holder: list[Exception] = []

    def _run():
        try:
            result["value"] = fn()
        except Exception as exc:
            exc_holder.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(timeout=2.0)
        if thread.is_alive():
            emit_fn("fit_progress", model=model_id, elapsed_s=round(time.perf_counter() - t0, 2))
    if exc_holder:
        raise exc_holder[0]
    return result["value"]


def fit_with_progress(model, X, y, model_id: str, emit_fn) -> None:
    """Fits `model` on (X, y), emitting periodic 'fit_progress' events via emit_fn.

    LightGBM and CatBoost expose real per-iteration callbacks, so those report
    actual completion fractions. Everything else (most sklearn ensembles) has no
    generic incremental-fit hook, so it falls back to run_with_heartbeat."""
    cls_name = type(model).__name__
    t0 = time.perf_counter()

    if cls_name in ("LGBMClassifier", "LGBMRegressor"):
        def lgbm_callback(env):
            total = env.end_iteration - env.begin_iteration
            step = max(1, total // 20)
            if env.iteration % step == 0 or env.iteration == env.end_iteration - 1:
                emit_fn("fit_progress", model=model_id, iteration=env.iteration + 1, total=total,
                        elapsed_s=round(time.perf_counter() - t0, 2))
        model.fit(X, y, callbacks=[lgbm_callback])
        return

    if cls_name in ("CatBoostClassifier", "CatBoostRegressor"):
        total = model.get_params().get("iterations") or 1000

        class _Callback:
            def after_iteration(self, info):
                step = max(1, total // 20)
                if info.iteration % step == 0 or info.iteration == total:
                    emit_fn("fit_progress", model=model_id, iteration=info.iteration, total=total,
                            elapsed_s=round(time.perf_counter() - t0, 2))
                return True

        model.fit(X, y, callbacks=[_Callback()])
        return

    run_with_heartbeat(lambda: model.fit(X, y), model_id, emit_fn)


def suggest_params(trial, search_space: dict) -> dict:
    params = {}
    for name, spec in search_space.items():
        t = spec["type"]
        if t == "int":
            kwargs = {"step": spec.get("step", 1)}
            if spec.get("log"):
                kwargs = {"log": True}
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], **kwargs)
        elif t == "float":
            kwargs = {"log": True} if spec.get("log") else {}
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], **kwargs)
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


def _classification_metrics(y_true, y_pred, model, X) -> dict:
    from sklearn.metrics import (
        accuracy_score, cohen_kappa_score, f1_score, matthews_corrcoef,
        precision_score, recall_score, roc_auc_score,
    )
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "Prec.": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    auc = 0.0
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            classes = model.classes_
            if len(classes) == 2:
                auc = roc_auc_score(y_true, proba[:, 1])
            else:
                auc = roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")
        except ValueError:
            auc = 0.0
    metrics["AUC"] = auc
    return {k: round(float(v), 4) for k, v in metrics.items()}


def _regression_metrics(y_true, y_pred) -> dict:
    import math
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }
    return {k: round(float(v), 4) for k, v in metrics.items()}


def main() -> None:
    args = json.loads(Path(sys.argv[1]).read_text())
    csv_path = args["csv_path"]
    feature_cols: list[str] = args["feature_cols"]
    label: str = args["label"]
    label_kind: str = args["label_kind"]  # "categorical" -> classification, "numeric" -> regression
    sample_frac: float = args.get("sample_frac", 0.1)
    fold: int = args.get("fold", 5)
    models_cfg: list[dict] = args["models"]  # [{model_id, n_trials, search_space}]
    models_dir = Path(args["models_dir"])
    run_id: str = args["run_id"]

    if label_kind not in ("categorical", "numeric"):
        emit("error", detail="label_kind must be 'categorical' or 'numeric'")
        return
    if not models_cfg:
        emit("error", detail="No models selected to tune")
        return

    _load_model_classes()

    import optuna
    import optuna.visualization as vis
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    is_classification = label_kind == "categorical"
    if is_classification:
        from pycaret.classification import ClassificationExperiment as Experiment
        scoring = "accuracy"
    else:
        from pycaret.regression import RegressionExperiment as Experiment
        scoring = "r2"

    df = pd.read_csv(csv_path)
    df = df[feature_cols + [label]]

    n_total = len(df)
    if is_classification:
        sample = df.groupby(label, group_keys=False).apply(
            lambda g: g.sample(frac=sample_frac, random_state=42)
        )
    else:
        sample = df.sample(frac=sample_frac, random_state=42)
    emit("sample_ready", n_total=n_total, n_sample=len(sample), sample_frac=sample_frac)

    # PyCaret's setup() only used here for its preprocessing pipeline (consistent
    # encoding/scaling with the rest of the app) — X/y below feed straight into
    # plain sklearn cross_val_score, not pycaret's own model functions.
    exp = Experiment()
    exp.setup(data=sample, target=label, session_id=42, train_size=0.8,
              n_jobs=1, verbose=False, html=False)
    X = exp.X_train_transformed
    y = exp.y_train_transformed

    studies: dict[str, object] = {}
    for cfg in models_cfg:
        mid = cfg["model_id"]
        n_trials = int(cfg.get("n_trials", 30))
        search_space = cfg.get("search_space", {})
        emit("search_start", model=mid, n_trials=n_trials)

        t0 = time.perf_counter()

        def objective(trial, mid=mid, search_space=search_space):
            params = suggest_params(trial, search_space)
            try:
                model = build_model(label_kind, mid, params)
                if is_classification:
                    cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
                else:
                    cv = KFold(n_splits=fold, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
                return float(scores.mean())
            except Exception:
                return 0.0 if is_classification else -1e9

        study = optuna.create_study(direction="maximize")

        def callback(study, trial, mid=mid, n_trials=n_trials, t0=t0):
            emit("trial_result", model=mid, trial=trial.number, n_trials=n_trials,
                 value=round(trial.value, 4) if trial.value is not None else None,
                 best_value=round(study.best_value, 4), params=trial.params,
                 elapsed_s=round(time.perf_counter() - t0, 2))

        try:
            study.optimize(objective, n_trials=n_trials, callbacks=[callback])
            studies[mid] = study
            history_json = vis.plot_optimization_history(study).to_json()
            slice_json = vis.plot_slice(study).to_json() if search_space else None
            emit("search_done", model=mid, best_value=round(study.best_value, 4),
                 best_params=study.best_params, history_json=history_json,
                 slice_json=slice_json, elapsed_s=round(time.perf_counter() - t0, 2))
        except Exception as exc:
            emit("search_done", model=mid, error=str(exc),
                 elapsed_s=round(time.perf_counter() - t0, 2))

    if not studies:
        emit("error", detail="Hyperparameter search failed for every selected model")
        return

    # ── Final fit: best hyperparameters, once, on the full dataset ──
    exp2 = Experiment()
    exp2.setup(data=df, target=label, session_id=42, train_size=0.8,
               n_jobs=1, verbose=False, html=False)
    emit("full_setup", n_train=len(exp2.X_train), n_test=len(exp2.X_test))

    Xtr, ytr = exp2.X_train_transformed, exp2.y_train_transformed
    Xte, yte = exp2.X_test_transformed, exp2.y_test_transformed

    fitted = {}
    for mid, study in studies.items():
        t0 = time.perf_counter()
        try:
            model = build_model(label_kind, mid, study.best_params)
            fit_with_progress(model, Xtr, ytr, mid, emit)
            fitted[mid] = model
            preds = model.predict(Xte)
            metrics = (_classification_metrics(yte, preds, model, Xte) if is_classification
                       else _regression_metrics(yte, preds))
            emit("full_refit_result", model=mid, metrics=metrics,
                 elapsed_s=round(time.perf_counter() - t0, 2))
        except Exception as exc:
            emit("full_refit_result", model=mid, error=str(exc),
                 elapsed_s=round(time.perf_counter() - t0, 2))

    # ── Save tuned models (this venv's joblib/sklearn version; same cross-env
    # loading caveat as automl_worker's saved models — not wired into /infer yet) ──
    models_dir.mkdir(exist_ok=True)
    saved = []
    for mid in fitted:
        filename = f"{run_id}__tuned_{mid}.joblib"
        joblib.dump(
            {"model": fitted[mid], "model_id": mid, "best_params": studies[mid].best_params},
            models_dir / filename,
        )
        saved.append({"model": mid, "filename": filename})
    emit("saved", saved_models=saved)

    emit("done")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit("error", detail=str(exc))
        raise
