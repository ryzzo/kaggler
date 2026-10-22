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
import math
import sys
import threading
import time
from pathlib import Path

import joblib
import pandas as pd

from gpu import describe as gpu_describe, gpu_kwargs, pycaret_use_gpu

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
    kwargs = {**FIXED_KWARGS[label_kind].get(model_id, {}), **gpu_kwargs(model_id), **params}
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

    # CatBoost's Python callbacks are CPU-only, so GPU-trained CatBoost falls
    # through to the heartbeat path below.
    if (cls_name in ("CatBoostClassifier", "CatBoostRegressor")
            and model.get_params().get("task_type") != "GPU"):
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


def stage_fractions(sample_frac: float, max_frac: float, stages: int) -> list[float]:
    """Sample fraction per stage, growing geometrically from sample_frac (coarse,
    cheap) to max_frac (fine, closer to the full data). One stage -> [sample_frac]."""
    if stages <= 1:
        return [sample_frac]
    hi = min(1.0, max(sample_frac, max_frac))
    return [round(sample_frac * (hi / sample_frac) ** (i / (stages - 1)), 4)
            for i in range(stages)]


def narrow_space(base_space: dict, trials, top_frac: float = 0.25, pad: float = 0.25) -> dict:
    """Next stage's search space, zoomed in on where the best trials landed.

    Takes the top `top_frac` of completed trials (at least 3) and, per parameter:
      - int/float: spans [min, max] of the top trials' values, widened by `pad` of
        that span (and never less than 5% of the original range, so the search
        can still move if the top trials happened to cluster tightly). Computed in
        log space for log-scale params. Always clamped to the ORIGINAL bounds.
      - categorical: keeps only the choices seen among the top trials.
    """
    done = [t for t in trials if t.value is not None]
    done.sort(key=lambda t: t.value, reverse=True)
    top = done[:max(3, math.ceil(len(done) * top_frac))]

    narrowed: dict = {}
    for name, spec in base_space.items():
        vals = [t.params[name] for t in top if name in t.params]
        if not vals:
            narrowed[name] = dict(spec)
            continue

        if spec["type"] == "categorical":
            keep = [c for c in spec["choices"] if c in vals]
            narrowed[name] = {**spec, "choices": keep or list(spec["choices"])}
            continue

        use_log = bool(spec.get("log"))
        fwd = math.log if use_log else (lambda v: float(v))
        inv = math.exp if use_log else (lambda v: v)
        full_lo, full_hi = fwd(spec["low"]), fwd(spec["high"])
        lo, hi = fwd(min(vals)), fwd(max(vals))
        margin = max((hi - lo) * pad, (full_hi - full_lo) * 0.05)
        new_lo = max(full_lo, lo - margin)
        new_hi = min(full_hi, hi + margin)

        if spec["type"] == "int":
            step = 1 if use_log else int(spec.get("step", 1))
            low = max(int(spec["low"]), int(math.floor(inv(new_lo))))
            high = min(int(spec["high"]), int(math.ceil(inv(new_hi))))
            if not use_log:  # keep `low` on the original step grid
                low = int(spec["low"]) + ((low - int(spec["low"])) // step) * step
            if high <= low:
                high = min(int(spec["high"]), low + step)
            narrowed[name] = {**spec, "low": low, "high": high}
        else:
            narrowed[name] = {**spec, "low": inv(new_lo), "high": inv(new_hi)}
    return narrowed


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

    # Coarse-to-fine search: n_trials (per model) applies to EACH stage; sample size
    # grows geometrically from sample_frac up to max_sample_frac across the stages.
    n_stages = max(1, min(6, int(args.get("stages", 1))))
    max_sample_frac: float = float(args.get("max_sample_frac", 0.5))
    top_frac: float = float(args.get("top_frac", 0.25))
    pad: float = float(args.get("pad", 0.25))

    # Full-data evaluation extras (see the end of main()).
    do_full_cv: bool = bool(args.get("full_cv", True))
    cv_folds: int = max(2, min(10, int(args.get("cv_folds", 5))))
    want_ensembles: bool = bool(args.get("ensembles", True))
    baseline_model = args.get("baseline_model")  # Data Training's best model id, default params

    if label_kind not in ("categorical", "numeric"):
        emit("error", detail="label_kind must be 'categorical' or 'numeric'")
        return
    if not models_cfg:
        emit("error", detail="No models selected to tune")
        return

    _load_model_classes()
    emit("gpu_info", **gpu_describe())

    import optuna
    import optuna.visualization as vis
    from sklearn.base import clone
    from sklearn.metrics import get_scorer
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

    def make_sample(frac: float):
        if is_classification:
            return df.groupby(label, group_keys=False).apply(
                lambda g: g.sample(frac=frac, random_state=42)
            )
        return df.sample(frac=frac, random_state=42)

    # ── Coarse-to-fine schedule ──
    # stages=1 is the classic single search. With stages>1 each later stage
    # searches a narrower space (around the best trials of the previous stage) on
    # a larger sample, warm-started from the previous stage's best.
    fracs = stage_fractions(sample_frac, max_sample_frac, n_stages)

    stage_data: dict[int, tuple] = {}

    def get_stage_data(i: int):
        """Sample + PyCaret preprocessing for stage i, computed once and shared
        by every model. PyCaret's setup() is only used for its preprocessing
        pipeline (consistent encoding/scaling with the rest of the app) — X/y feed
        straight into plain sklearn cross_val_score, not pycaret's model functions."""
        if i not in stage_data:
            sample = make_sample(fracs[i])
            exp = Experiment()
            exp.setup(data=sample, target=label, session_id=42, train_size=0.8,
                      n_jobs=1, verbose=False, html=False, use_gpu=pycaret_use_gpu())
            stage_data[i] = (exp.X_train_transformed, exp.y_train_transformed, len(sample))
        return stage_data[i]

    first_n = get_stage_data(0)[2]
    emit("sample_ready", n_total=n_total, n_sample=first_n, sample_frac=fracs[0],
         stages=n_stages, stage_fractions=fracs)

    studies: dict[str, object] = {}
    for cfg in models_cfg:
        mid = cfg["model_id"]
        n_trials = int(cfg.get("n_trials", 30))  # per stage
        base_space = cfg.get("search_space", {})
        total_trials = n_trials * n_stages
        emit("search_start", model=mid, n_trials=total_trials, stages=n_stages,
             trials_per_stage=n_trials)

        t0 = time.perf_counter()
        last_study = None
        stage_summaries: list[dict] = []
        space = base_space
        error: Exception | None = None

        for si in range(n_stages):
            try:
                X, y, n_sample = get_stage_data(si)
                offset = si * n_trials
                emit("stage_start", model=mid, stage=si + 1, stages=n_stages,
                     sample_frac=fracs[si], n_sample=n_sample, search_space=space)

                def objective(trial, mid=mid, space=space, X=X, y=y):
                    params = suggest_params(trial, space)
                    emit("trial_start", model=mid, trial=offset + trial.number, stage=si + 1, params=params)
                    try:
                        model = build_model(label_kind, mid, params)
                        if is_classification:
                            cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
                        else:
                            cv = KFold(n_splits=fold, shuffle=True, random_state=42)
                        # Same result as cross_val_score(..., n_jobs=1), but one fold at a
                        # time so each fold's score can be streamed to the UI as it lands.
                        scorer = get_scorer(scoring)
                        take = (lambda a, i: a.iloc[i]) if hasattr(X, "iloc") else (lambda a, i: a[i])
                        fold_scores = []
                        for k, (tr, te) in enumerate(cv.split(X, y)):
                            tf = time.perf_counter()
                            m = clone(model)
                            m.fit(take(X, tr), take(y, tr))
                            fold_scores.append(float(scorer(m, take(X, te), take(y, te))))
                            emit("fold_result", model=mid, trial=offset + trial.number, fold=k + 1, folds=fold,
                                 metrics={scoring: round(fold_scores[-1], 4)},
                                 running={scoring: round(sum(fold_scores) / len(fold_scores), 4)},
                                 elapsed_s=round(time.perf_counter() - tf, 2))
                        return float(sum(fold_scores) / len(fold_scores))
                    except Exception:
                        return 0.0 if is_classification else -1e9

                study = optuna.create_study(direction="maximize")
                if last_study is not None:
                    # Warm start: re-score the previous stage's best on this stage's
                    # (larger) sample as its first trial.
                    try:
                        study.enqueue_trial(last_study.best_params)
                    except Exception:
                        pass

                def callback(study, trial, mid=mid, si=si, offset=offset, t0=t0,
                             total_trials=total_trials):
                    emit("trial_result", model=mid, trial=offset + trial.number,
                         n_trials=total_trials, stage=si + 1, stages=n_stages,
                         value=round(trial.value, 4) if trial.value is not None else None,
                         best_value=round(study.best_value, 4), params=trial.params,
                         elapsed_s=round(time.perf_counter() - t0, 2))

                study.optimize(objective, n_trials=n_trials, callbacks=[callback])
                last_study = study
                stage_summaries.append({
                    "stage": si + 1, "sample_frac": fracs[si], "n_sample": n_sample,
                    "best_value": round(study.best_value, 4),
                    "best_params": study.best_params, "search_space": space,
                })
                emit("stage_done", model=mid, stages=n_stages, **stage_summaries[-1])
                if si + 1 < n_stages:
                    space = narrow_space(base_space, study.trials, top_frac, pad)
            except Exception as exc:
                error = exc
                break  # keep whatever earlier stages produced

        if last_study is None:
            emit("search_done", model=mid, error=str(error) if error else "Search failed",
                 elapsed_s=round(time.perf_counter() - t0, 2))
            continue
        try:
            studies[mid] = last_study
            history_json = vis.plot_optimization_history(last_study).to_json()
            slice_json = vis.plot_slice(last_study).to_json() if space else None
            emit("search_done", model=mid, best_value=round(last_study.best_value, 4),
                 best_params=last_study.best_params, history_json=history_json,
                 slice_json=slice_json, stage_summaries=stage_summaries,
                 warning=(f"Stopped after stage {len(stage_summaries)}/{n_stages}: {error}"
                          if error else None),
                 elapsed_s=round(time.perf_counter() - t0, 2))
        except Exception as exc:
            emit("search_done", model=mid, error=str(exc),
                 elapsed_s=round(time.perf_counter() - t0, 2))

    if not studies:
        emit("error", detail="Hyperparameter search failed for every selected model")
        return

    # ── Final fit: best hyperparameters, once, on the full dataset ──
    exp2 = Experiment()
    exp2.setup(data=df, target=label, session_id=42, train_size=0.8,
               n_jobs=1, verbose=False, html=False, use_gpu=pycaret_use_gpu())
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

    # ── Full-data evaluation extras ─────────────────────────────────────────
    # 1. a default-hyperparameter baseline (so "tuned vs default" is comparable),
    # 2. k-fold cross-validation over ALL rows (train + hold-out) for each tuned
    #    model — a steadier "performance on the full dataset" than the single
    #    hold-out split, and
    # 3. ensembles of the tuned models (voting, stacking).
    Xall = pd.concat([Xtr, Xte])
    yall = pd.concat([ytr, yte])

    def cv_heartbeat(_event, **kw):
        emit("cv_progress", **kw)

    def full_data_cv(key: str, make_model) -> dict:
        t0 = time.perf_counter()
        cv = (StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if is_classification
              else KFold(n_splits=cv_folds, shuffle=True, random_state=42))
        scores = run_with_heartbeat(
            lambda: cross_val_score(make_model(), Xall, yall, cv=cv, scoring=scoring, n_jobs=1),
            key, cv_heartbeat)
        return {"mean": round(float(scores.mean()), 4), "std": round(float(scores.std()), 4),
                "folds": cv_folds, "scoring": scoring, "n_rows": len(Xall),
                "elapsed_s": round(time.perf_counter() - t0, 2)}

    # 1. baseline (Data Training's best model type, default hyperparameters)
    baseline_ok = False
    if baseline_model and baseline_model in MODEL_CLASS_MAP[label_kind]:
        t0 = time.perf_counter()
        emit("baseline_start", model=baseline_model)
        try:
            base = build_model(label_kind, baseline_model, {})
            fit_with_progress(base, Xtr, ytr, baseline_model, emit)
            preds = base.predict(Xte)
            metrics = (_classification_metrics(yte, preds, base, Xte) if is_classification
                       else _regression_metrics(yte, preds))
            baseline_ok = True
            emit("baseline_result", model=baseline_model, metrics=metrics,
                 elapsed_s=round(time.perf_counter() - t0, 2))
        except Exception as exc:
            emit("baseline_result", model=baseline_model, error=str(exc),
                 elapsed_s=round(time.perf_counter() - t0, 2))

    # 2. full-data cross-validation
    if do_full_cv:
        cv_targets = [(mid, False, (lambda mid=mid: build_model(label_kind, mid, studies[mid].best_params)))
                      for mid in fitted]
        if baseline_ok:
            cv_targets.append((baseline_model, True, lambda: build_model(label_kind, baseline_model, {})))
        for key, is_base, make in cv_targets:
            emit("full_cv_start", model=key, baseline=is_base, folds=cv_folds)
            try:
                emit("full_cv_result", model=key, baseline=is_base, **full_data_cv(key, make))
            except Exception as exc:
                emit("full_cv_result", model=key, baseline=is_base, error=str(exc))

    # 3. ensembles of the tuned models
    if want_ensembles and len(fitted) >= 2:
        member_ids = list(fitted)

        def make_ensemble(kind: str):
            members = [(mid, build_model(label_kind, mid, studies[mid].best_params)) for mid in member_ids]
            if is_classification:
                from sklearn.ensemble import StackingClassifier, VotingClassifier
                from sklearn.linear_model import LogisticRegression
                if kind == "ens_vote":
                    soft = all(hasattr(m, "predict_proba") for _, m in members)
                    return VotingClassifier(members, voting="soft" if soft else "hard")
                return StackingClassifier(members, final_estimator=LogisticRegression(max_iter=1000), cv=3)
            from sklearn.ensemble import StackingRegressor, VotingRegressor
            from sklearn.linear_model import RidgeCV
            if kind == "ens_vote":
                return VotingRegressor(members)
            return StackingRegressor(members, final_estimator=RidgeCV(), cv=3)

        for kind in ("ens_vote", "ens_stack"):
            t0 = time.perf_counter()
            emit("ensemble_start", ensemble=kind, members=member_ids)
            try:
                ens = make_ensemble(kind)
                run_with_heartbeat(lambda: ens.fit(Xtr, ytr), kind, emit)
                preds = ens.predict(Xte)
                metrics = (_classification_metrics(yte, preds, ens, Xte) if is_classification
                           else _regression_metrics(yte, preds))
                filename = f"{run_id}__ensemble_{kind}.joblib"
                joblib.dump({"model": ens, "model_id": kind, "members": member_ids, "best_params": {}},
                            models_dir / filename)
                emit("ensemble_result", ensemble=kind, members=member_ids, metrics=metrics,
                     filename=filename, elapsed_s=round(time.perf_counter() - t0, 2),
                     voting=getattr(ens, "voting", None))
            except Exception as exc:
                emit("ensemble_result", ensemble=kind, members=member_ids, error=str(exc),
                     elapsed_s=round(time.perf_counter() - t0, 2))
                continue
            # Stacking already cross-validates internally; a CV-of-a-CV is too slow to be useful.
            if do_full_cv and kind == "ens_vote":
                emit("full_cv_start", model=kind, baseline=False, folds=cv_folds)
                try:
                    emit("full_cv_result", model=kind, baseline=False, **full_data_cv(kind, lambda: make_ensemble(kind)))
                except Exception as exc:
                    emit("full_cv_result", model=kind, baseline=False, error=str(exc))

    emit("done")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit("error", detail=str(exc))
        raise
