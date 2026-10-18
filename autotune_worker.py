"""
Runs inside .venv-automl as a subprocess launched by routers/autopilot.py.
Never imported by the main app — invoked via `python autotune_worker.py <args-json-file>`.

"Autopilot": an endless hyperparameter search that runs until it is told to stop.
It climbs a ladder from cheap/simple searches to slow/complex ones, using more and
more of the training data each round, and keeps the best model found so far saved
under a stable filename so Inference can always use "the current best".

  rounds 0-5   a fixed ladder: complexity level 1 -> 5, search data 5% -> 75%,
               more trials each time (see LADDER)
  rounds 6+    level 6 (widest spaces) on 100% of the training data, forever,
               alternating EXPLORE rounds (wide space, fresh sampler seed) with
               REFINE rounds (zoom in around the best parameters found so far,
               tighter every time), with the trial count growing each round

Fair comparison: every round shares ONE PyCaret pipeline + train/test split built
once from the full clean data (same session_id/train_size as infer_worker.py, so
the saved model works with the existing inference path). A model's search runs on
a nested, stratified subset of the training split; its best parameters are then
refit on the FULL training split and scored on the fixed hold-out. That hold-out
score is what decides whether a candidate becomes the new best.

Stop: the parent creates `stop_file`; it is checked between trials, so a stop
finishes the current trial (not the whole round) and exits cleanly.
State (round counter, per-model bests, global best) is persisted to `state_path`
after every model, so a stopped or crashed run resumes where it left off.

Emits one JSON object per line to stdout (forwarded as SSE by the parent).
"""
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gpu import describe as gpu_describe, pycaret_use_gpu
from tune_worker import (
    MODEL_CLASS_MAP, _classification_metrics, _load_model_classes, _regression_metrics,
    build_model, fit_with_progress, suggest_params,
)


def emit(event: str, **data) -> None:
    print(json.dumps({"event": event, **data}), flush=True)


# ── Search spaces ────────────────────────────────────────────────────────────
# SPACES[model][param] = {first_level: spec, ...}. At complexity level L a param
# uses the spec with the greatest key <= L; a param with no key <= L is not
# searched yet (it enters the space at its first key). So low levels search few
# parameters in tight ranges and high levels search many in wide ranges.

def P(kind, lo=None, hi=None, log=False, step=None, choices=None) -> dict:
    if kind == "categorical":
        return {"type": "categorical", "choices": choices}
    spec = {"type": kind, "low": lo, "high": hi}
    if log:
        spec["log"] = True
    elif step:
        spec["step"] = step
    return spec


SPACES: dict[str, dict[str, dict[int, dict]]] = {
    # ── linear / simple ──
    "lr": {  # LogisticRegression (classification only; regression "lr" has no params)
        "C": {1: P("float", 1e-3, 1e2, log=True), 3: P("float", 1e-5, 1e3, log=True)},
    },
    "ridge": {
        "alpha": {1: P("float", 1e-2, 1e2, log=True), 3: P("float", 1e-4, 1e4, log=True)},
    },
    "lasso": {
        "alpha": {1: P("float", 1e-4, 1e1, log=True), 3: P("float", 1e-6, 1e2, log=True)},
    },
    "en": {
        "alpha": {1: P("float", 1e-4, 1e1, log=True), 3: P("float", 1e-6, 1e2, log=True)},
        "l1_ratio": {1: P("float", 0.1, 0.9), 3: P("float", 0.0, 1.0)},
    },
    "dt": {
        "max_depth": {1: P("int", 2, 12), 2: P("int", 2, 30)},
        "min_samples_leaf": {1: P("int", 1, 20), 3: P("int", 1, 50)},
        "min_samples_split": {2: P("int", 2, 40)},
    },
    "knn": {
        "n_neighbors": {1: P("int", 3, 25), 2: P("int", 1, 50)},
        "weights": {2: P("categorical", choices=["uniform", "distance"])},
        "p": {2: P("int", 1, 2)},
    },
    "svm": {  # SVR (regression only) — O(n^2), so it never runs past level 2
        "C": {1: P("float", 1e-1, 1e2, log=True)},
        "epsilon": {2: P("float", 1e-3, 1.0, log=True)},
    },
    # ── bagged trees ──
    "rf": {
        "n_estimators": {1: P("int", 50, 200, step=50), 3: P("int", 100, 600, step=50),
                         5: P("int", 200, 1500, step=100)},
        "max_depth": {1: P("int", 4, 16), 4: P("int", 3, 40)},
        "min_samples_leaf": {2: P("int", 1, 10), 5: P("int", 1, 30)},
        "min_samples_split": {3: P("int", 2, 20)},
        "max_features": {3: P("float", 0.3, 1.0), 5: P("float", 0.1, 1.0)},
    },
    "et": {
        "n_estimators": {1: P("int", 50, 200, step=50), 3: P("int", 100, 600, step=50),
                         5: P("int", 200, 1500, step=100)},
        "max_depth": {1: P("int", 4, 16), 4: P("int", 3, 40)},
        "min_samples_leaf": {2: P("int", 1, 10), 5: P("int", 1, 30)},
        "min_samples_split": {3: P("int", 2, 20)},
        "max_features": {3: P("float", 0.3, 1.0), 5: P("float", 0.1, 1.0)},
    },
    # ── boosting ──
    "gbc": {
        "n_estimators": {3: P("int", 50, 300, step=50), 5: P("int", 100, 600, step=50)},
        "learning_rate": {3: P("float", 0.03, 0.3, log=True), 5: P("float", 0.01, 0.3, log=True)},
        "max_depth": {3: P("int", 2, 6), 5: P("int", 2, 10)},
        "subsample": {4: P("float", 0.6, 1.0)},
        "min_samples_leaf": {5: P("int", 1, 30)},
    },
    "lightgbm": {
        "n_estimators": {2: P("int", 50, 300, step=50), 4: P("int", 100, 1000, step=50),
                         5: P("int", 200, 3000, step=100)},
        "learning_rate": {2: P("float", 0.03, 0.3, log=True), 4: P("float", 0.01, 0.3, log=True),
                          5: P("float", 0.003, 0.3, log=True)},
        "num_leaves": {2: P("int", 8, 64), 4: P("int", 8, 256), 5: P("int", 4, 512)},
        "max_depth": {3: P("int", 3, 12), 5: P("int", 3, 20)},
        "min_child_samples": {3: P("int", 5, 60), 5: P("int", 2, 120)},
        "subsample": {3: P("float", 0.6, 1.0), 5: P("float", 0.4, 1.0)},
        "colsample_bytree": {3: P("float", 0.6, 1.0), 5: P("float", 0.3, 1.0)},
        "reg_alpha": {4: P("float", 1e-8, 10.0, log=True)},
        "reg_lambda": {4: P("float", 1e-8, 10.0, log=True)},
        "min_split_gain": {6: P("float", 0.0, 1.0)},
    },
    "catboost": {
        "iterations": {2: P("int", 100, 500, step=50), 4: P("int", 200, 1500, step=100),
                       5: P("int", 300, 4000, step=100)},
        "learning_rate": {2: P("float", 0.03, 0.3, log=True), 4: P("float", 0.01, 0.3, log=True),
                          5: P("float", 0.005, 0.3, log=True)},
        "depth": {2: P("int", 4, 8), 4: P("int", 3, 10)},
        "l2_leaf_reg": {3: P("float", 1.0, 10.0), 4: P("float", 1e-2, 30.0, log=True)},
        "random_strength": {5: P("float", 1e-3, 10.0, log=True)},
        "border_count": {5: P("int", 32, 254)},
    },
}

# Slow models stop being searched past this level (their later spaces add cost, not value).
MAX_LEVEL = {"knn": 2, "svm": 2, "lr": 3, "ridge": 3, "lasso": 3, "en": 3, "dt": 3,
             "gbc": 5, "gbr": 5}

# Always-on estimator kwargs for autopilot (stored with the saved model's best_params
# so the existing inference path rebuilds an identical estimator if it ever has to).
AUTO_FIXED = {
    "lightgbm": {"subsample_freq": 1},   # without this LightGBM ignores `subsample`
    "rf": {"n_jobs": -1}, "et": {"n_jobs": -1}, "knn": {"n_jobs": -1},
}

# Task-specific aliases: the regression "gbc" slot is "gbr".
ALIASES = {"gbr": "gbc"}


def space_for(model_id: str, level: int) -> dict:
    table = SPACES.get(ALIASES.get(model_id, model_id))
    if not table:
        return {}
    if level > MAX_LEVEL.get(model_id, 99):
        return {}
    space = {}
    for name, by_level in table.items():
        keys = [k for k in by_level if k <= level]
        if keys:
            space[name] = dict(by_level[max(keys)])
    return space


# ── Round schedule ───────────────────────────────────────────────────────────
# (complexity level, fraction of training split searched on, trials per model)
LADDER = [
    (1, 0.05, 15),
    (1, 0.10, 20),
    (2, 0.20, 30),
    (3, 0.35, 40),
    (4, 0.50, 60),
    (5, 0.75, 80),
]


def plan_round(r: int) -> dict:
    if r < len(LADDER):
        level, frac, trials = LADDER[r]
        return {"round": r, "mode": "ladder", "level": level, "frac": frac, "trials": trials}
    k = r - len(LADDER)  # 0, 1, 2, ... after the ladder
    return {
        "round": r,
        "mode": "explore" if k % 2 == 0 else "refine",
        "refine_step": k // 2,
        "level": 6, "frac": 1.0,
        "trials": min(400, 100 + 20 * k),
    }


def zoom_space(space: dict, center: dict, refine_step: int) -> dict:
    """REFINE rounds: shrink each numeric range around the best known value
    (tighter with every refine step), clamped to the original bounds. Categoricals
    keep all their choices (the warm-start trial already carries the best one)."""
    factor = max(1.1, 1.6 * (0.9 ** refine_step))  # multiplicative window for log params
    frac = max(0.03, 0.15 * (0.9 ** refine_step))   # +/- share of range for linear params
    out = {}
    for name, spec in space.items():
        c = center.get(name)
        if spec["type"] == "categorical" or c is None:
            out[name] = dict(spec)
            continue
        lo, hi = spec["low"], spec["high"]
        if spec.get("log"):
            nlo, nhi = max(lo, c / factor), min(hi, c * factor)
        else:
            w = (hi - lo) * frac
            nlo, nhi = max(lo, c - w), min(hi, c + w)
        if spec["type"] == "int":
            nlo, nhi = int(math.floor(nlo)), int(math.ceil(nhi))
            step = spec.get("step", 1)
            if not spec.get("log") and step > 1:
                nlo = lo + ((nlo - lo) // step) * step                # snap down onto the grid
                nhi = min(hi, lo + math.ceil((nhi - lo) / step) * step)  # snap up onto the grid
            if nhi <= nlo:
                nhi = min(hi, nlo + step)
        out[name] = {**spec, "low": nlo, "high": nhi}
    return out


def fit_to_space(params: dict, space: dict) -> dict:
    """Subset of a known-good parameter set that is valid inside `space` (used to
    warm-start a study without ever enqueueing an out-of-range value)."""
    ok = {}
    for name, spec in space.items():
        if name not in params:
            continue
        v = params[name]
        if spec["type"] == "categorical":
            if v in spec["choices"]:
                ok[name] = v
        elif spec["low"] <= v <= spec["high"]:
            ok[name] = v
    return ok


# ── Data ─────────────────────────────────────────────────────────────────────
def nested_order(y, classification: bool, seed: int = 42) -> np.ndarray:
    """A row ordering whose every prefix is a (stratified) random subset, so the
    5% search sample is contained in the 10% one, and so on up to 100%."""
    rng = np.random.RandomState(seed)
    y_arr = np.asarray(y)
    key = np.empty(len(y_arr))
    if classification:
        for c in np.unique(y_arr):
            idx = np.where(y_arr == c)[0]
            rng.shuffle(idx)
            key[idx] = (np.arange(len(idx)) + rng.rand()) / len(idx)
    else:
        key = rng.rand(len(y_arr))
    return np.argsort(key, kind="stable")


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def main() -> None:
    warnings.filterwarnings("ignore")  # optuna experimental/sklearn convergence noise
    args = json.loads(Path(sys.argv[1]).read_text())
    csv_path = args["csv_path"]
    feature_cols: list[str] = args["feature_cols"]
    label: str = args["label"]
    label_kind: str = args["label_kind"]
    fold: int = int(args.get("fold", 3))
    wanted: list[str] | None = args.get("models") or None
    models_dir = Path(args["models_dir"])
    best_path = Path(args["best_path"])
    state_path = Path(args["state_path"])
    stop_file = Path(args["stop_file"])
    min_gain: float = float(args.get("min_gain", 1e-6))

    if label_kind not in ("categorical", "numeric"):
        emit("error", detail="label_kind must be 'categorical' or 'numeric'")
        return
    is_cls = label_kind == "categorical"

    def stopping() -> bool:
        return stop_file.exists()

    _load_model_classes()
    classes = MODEL_CLASS_MAP[label_kind]
    candidates = [m for m in (wanted or list(classes)) if m in classes and SPACES.get(ALIASES.get(m, m))]
    if is_cls:
        candidates = [m for m in candidates if m != "gbr"]
    else:
        candidates = [m for m in candidates if m != "gbc"]
    # regression "lr" (LinearRegression) has no hyperparameters; classification "lr" does
    if not is_cls:
        candidates = [m for m in candidates if m != "lr"]
    if not candidates:
        emit("error", detail="None of the selected models has a tunable search space for autopilot")
        return

    import optuna
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if is_cls:
        from pycaret.classification import ClassificationExperiment as Experiment
        scoring, primary = "accuracy", "Accuracy"
    else:
        from pycaret.regression import RegressionExperiment as Experiment
        scoring, primary = "r2", "R2"

    # ── One shared pipeline + split for every round (matches infer_worker.py) ──
    emit("pipeline_start")
    df = pd.read_csv(csv_path)[feature_cols + [label]]
    exp = Experiment()
    exp.setup(data=df, target=label, session_id=42, train_size=0.8,
              n_jobs=1, verbose=False, html=False, use_gpu=pycaret_use_gpu())
    Xtr, ytr = exp.X_train_transformed, exp.y_train_transformed
    Xte, yte = exp.X_test_transformed, exp.y_test_transformed
    order = nested_order(ytr, is_cls)
    n_train = len(Xtr)
    emit("pipeline_ready", n_train=n_train, n_test=len(Xte), gpu=gpu_describe(), models=candidates)

    # ── Resume ──
    state = {"round": 0, "best": None, "models": {}}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            emit("resumed", round=state["round"], best=state.get("best"))
        except Exception:
            pass

    def save_state():
        write_json_atomic(state_path, state)

    def score_of(metrics: dict) -> float:
        return float(metrics[primary])

    def fixed(mid: str, params: dict) -> dict:
        return {**AUTO_FIXED.get(mid, {}), **params}

    r = int(state["round"])
    while not stopping():
        plan = plan_round(r)
        level, frac, n_trials = plan["level"], plan["frac"], plan["trials"]
        n_rows = max(min(n_train, 200), int(round(frac * n_train)))
        pos = order[:n_rows]
        Xs, ys = Xtr.iloc[pos], ytr.iloc[pos]
        round_models = [m for m in candidates if space_for(m, level)]
        if not round_models and level >= 6:
            # Only simple models were selected: past the ladder they keep getting
            # re-searched (fresh seeds / tighter zooms) at their own top complexity.
            round_models = list(candidates)
        emit("round_start", **plan, n_rows=n_rows, n_train=n_train, models=round_models)
        t_round = time.perf_counter()
        board = []

        for mi, mid in enumerate(round_models):
            if stopping():
                break
            known = state["models"].get(mid, {})
            space = space_for(mid, min(level, MAX_LEVEL.get(mid, 99)))
            if plan["mode"] == "refine" and known.get("params"):
                space = zoom_space(space, known["params"], plan["refine_step"])
            emit("model_start", model=mid, round=r, level=level, n_params=len(space),
                 search_space=space)
            t0 = time.perf_counter()

            def objective(trial, mid=mid, space=space):
                params = suggest_params(trial, space)
                try:
                    model = build_model(label_kind, mid, fixed(mid, params))
                    cv = (StratifiedKFold(n_splits=fold, shuffle=True, random_state=42) if is_cls
                          else KFold(n_splits=fold, shuffle=True, random_state=42))
                    return float(cross_val_score(model, Xs, ys, cv=cv, scoring=scoring, n_jobs=1).mean())
                except Exception:
                    return 0.0 if is_cls else -1e9

            sampler = optuna.samplers.TPESampler(seed=r * 1000 + mi, multivariate=True)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            if known.get("params"):
                warm = fit_to_space(known["params"], space)
                if warm:
                    try:
                        study.enqueue_trial(warm)  # re-score the best known config at this scale first
                    except Exception:
                        pass

            def callback(study, trial, mid=mid, t0=t0, n_trials=n_trials):
                emit("trial_result", model=mid, round=r, trial=trial.number, n_trials=n_trials,
                     value=round(trial.value, 4) if trial.value is not None else None,
                     best_value=round(study.best_value, 4), params=trial.params,
                     elapsed_s=round(time.perf_counter() - t0, 2))
                if stopping():
                    study.stop()

            try:
                study.optimize(objective, n_trials=n_trials, callbacks=[callback])
            except Exception as exc:
                emit("model_result", model=mid, round=r, error=str(exc))
                continue
            if not study.trials or stopping():
                break

            # Refit the best parameters on the FULL training split, score on the hold-out.
            best_params = fixed(mid, study.best_params)
            try:
                model = build_model(label_kind, mid, best_params)
                fit_with_progress(model, Xtr, ytr, mid, emit)
                preds = model.predict(Xte)
                metrics = (_classification_metrics(yte, preds, model, Xte) if is_cls
                           else _regression_metrics(yte, preds))
            except Exception as exc:
                emit("model_result", model=mid, round=r, error=str(exc),
                     elapsed_s=round(time.perf_counter() - t0, 2))
                continue

            score = score_of(metrics)
            emit("model_result", model=mid, round=r, search_cv=round(study.best_value, 4),
                 score=round(score, 4), metrics=metrics, best_params=best_params,
                 elapsed_s=round(time.perf_counter() - t0, 2))
            board.append({"model": mid, "score": round(score, 4)})

            # Per-model memory (warm starts) — keep the best test score seen for this model.
            if score > known.get("score", -math.inf):
                state["models"][mid] = {"score": score, "params": best_params}

            cur = state.get("best")
            if cur is None or score > cur["score"] + min_gain:
                models_dir.mkdir(exist_ok=True)
                tmp = best_path.with_suffix(".joblib.tmp")
                joblib.dump({"model": model, "model_id": mid, "best_params": best_params,
                             "score": score, "round": r}, tmp)
                os.replace(tmp, best_path)  # atomic: inference never sees a half-written file
                state["best"] = {
                    "model": mid, "score": score, "metric": primary, "metrics": metrics,
                    "best_params": best_params, "round": r, "level": level, "frac": frac,
                    "filename": best_path.name, "saved_at": time.time(),
                }
                emit("new_best", **state["best"])
            save_state()

        if stopping():
            break
        state["round"] = r + 1
        save_state()
        emit("round_done", round=r, mode=plan["mode"], level=level, frac=frac,
             elapsed_s=round(time.perf_counter() - t_round, 2), board=board,
             best=state.get("best"))
        r += 1

    save_state()
    emit("stopped", rounds_completed=state["round"], best=state.get("best"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit("error", detail=str(exc))
        raise
