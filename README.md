# Kaggler — End-to-End ML Pipeline

A browser-based machine learning pipeline designed for Kaggle competitions. Upload a CSV, explore it, clean it, let AutoML scout models, tune the one you like with Optuna, and generate a submission file — all through a dark-themed web UI backed by a FastAPI server.

---

## Pipeline Overview

```
Data Collection → Data Exploration → Data Preparation → Data Training → Hyperparameter Search → Inference
```

Each stage is a separate page with a navigation bar and a "Next →" banner that carries context forward automatically via `sessionStorage`.

---

## Techniques & Methods

### 1. Data Collection

- Drag-and-drop CSV upload with batch support.
- Files are stored temporarily in `uploads/`; only the most recent upload is kept per session.

---

### 2. Data Exploration

**Summary statistics**
- Per-column: mean, median, std, min, max, unique count, missing count and percentage.
- Auto-detection of column type (numeric vs. categorical) based on parsability.

**Visualisations**
- Histogram for numeric columns.
- Bar chart of top-20 value counts for categorical columns.
- Pairwise correlation heatmap (Pearson r) for all numeric features — rendered on a `<canvas>` element with a diverging blue→red colour scale.
- Feature-vs-label correlation bar chart, sorted by |r|.

**Missingness correlation**
- For every column with at least one missing value, a binary indicator vector (1 = missing, 0 = present) is computed.
- That indicator is correlated (Pearson r) with every numeric column's values and with every other column's missingness indicator.
- Results are sorted by |r| and displayed as a bar chart per column, labelled *weak* (|r| < 0.4), *moderate* (0.4–0.7), or *strong* (≥ 0.7).
- High |r| signals that missingness is **not at random (MNAR/MAR)** and guides the imputation strategy chosen in the next step.

---

### 3. Data Preparation

#### Missing data imputation

Strategy is chosen automatically per column based on missing percentage and missingness-correlation strength:

| Condition | Method |
|---|---|
| Numeric, < 5% missing, weak missingness correlation (max \|r\| < 0.4) | **Median fill** — fills with the column median, robust to outliers |
| Numeric, < 5% missing, strong missingness correlation | **KNN imputation** — missingness is not random so a simple statistic is insufficient |
| Categorical, < 5% missing | **Mode fill** — fills with the most frequent category |
| Any type, 5 – 30% missing | **KNN imputation** (see below) |
| Any type, > 30% missing | **Skipped** — blanks are later encoded as a dedicated `__missing__` category |

**KNN imputation detail**
- Donor rows (rows where the target column is present) are collected.
- Up to **5 000 donors** are sampled; if more exist they are stride-sampled after sorting by the first feature column so the sample is well-distributed across the value range.
- Features used: numeric columns with < 5% missing (excluding the target column itself).
- `sklearn.neighbors.KNeighborsRegressor` (k = 5) for numeric targets; `KNeighborsClassifier` (k = 5) for categorical targets.
- The **fitted model, feature column list, and column means** (used as fallback for any missing feature values at inference) are serialised with `joblib` and stored alongside the prepared data.

**Imputation params persistence**
- Saved to `prepared/{id}__imputation.joblib`.
- `imputation_params_id` is returned by `/prepare` and carried in `sessionStorage`.

#### Feature encoding & scaling (applied after imputation)

| Column type | Transform |
|---|---|
| Categorical | **Label encoding** — each unique string is mapped to an integer code. The mapping is persisted and inverted at inference to restore original labels. |
| Numeric | **Standard scaling (z-score)** — `(x − μ) / σ`. Mean and std are persisted and used to inverse-transform regression predictions back to the original unit. |

**Output files**
- `prepared/{id}.csv` — encoded + scaled dataset.
- `prepared/{id}__clean.csv` — imputed but *not* transformed dataset (original scale). This is the file Data Training, Hyperparameter Search, and Inference all work from — PyCaret does its own encoding/scaling internally, so it needs unencoded features.

---

### 4. Data Training — PyCaret AutoML (model scouting)

Data Training's job is **scouting, not training**: compare a broad set of candidate models on a small sample and surface which ones are worth pursuing further. Actual training happens on the Hyperparameter Search page.

#### Task detection
The label column's presence in `encoding_info` determines **classification** vs **regression**; PyCaret's `ClassificationExperiment` or `RegressionExperiment` is used accordingly.

#### Sampling & comparison
- A **10% sample** is taken — stratified by label for classification, plain random for regression.
- ~15–20 candidate models are compared via 5-fold cross-validation on that sample (a turbo-safe subset per task, e.g. excludes known-slow ones like kernel SVMs, ARD, TheilSen).

#### Ranking (tie-break cascade)
Models aren't ranked by a single metric — ties fall through a cascade, only moving to the next metric when everything before it is exactly equal:

| Task | Cascade |
|---|---|
| Classification | Accuracy → F1 → MCC → Kappa → AUC |
| Regression | R² → RMSE → MAE → MSE (lower-is-better metrics sign-flipped internally) |

The same cascade is mirrored client-side so the "best" badge shown live during comparison always matches what the backend ultimately picks.

#### ID column exclusion
Columns matching `^id$`, `^id[_-]`, or `[_-]id$` (case-insensitive) are excluded from features everywhere in the pipeline — Data Training, Hyperparameter Search, and Inference all derive features from the full column list server-side using this same rule, so a saved model's feature count can never silently drift.

#### Best-model auto-save
Once ranked, the **single best model** (default hyperparameters) is fit on the **full dataset** and saved to `models/{run_id}__automl_{model}.joblib` — so Hyperparameter Search and Inference have a ready-to-use fit for the default candidate instead of training from scratch the first time it's needed.

#### Streaming progress
Results stream to the browser via SSE: `sample_ready → compare_start → compare_result` (per model) `→ leaderboard_done → best_model_fit_start → fit_progress → best_model_saved → done`.

---

### 5. Hyperparameter Search — Optuna

Pick one or more of Data Training's top-3 models and tune them with a fully user-configurable search space.

#### Search space configuration
For each selected model, a config card lets you:
- Enable/disable individual hyperparameters (seeded with sensible defaults per model, e.g. `n_estimators`, `max_depth`, `learning_rate`, `max_features`).
- Edit numeric ranges (low/high/step) and toggle log-scale sampling.
- Choose which categorical options to include.
- Set the number of trials.

#### Search execution
- Uses **raw Optuna** driving plain scikit-learn/LightGBM/CatBoost estimators directly (not PyCaret's own `tune_model`) — this is what makes a fully custom, per-parameter search space possible.
- PyCaret is still used, but only for its `setup()` preprocessing pipeline, so encoding/scaling stays consistent with the rest of the app.
- Search runs cross-validated on the same 10% sample Data Training used (optimizing accuracy or R² depending on task).
- The best-found hyperparameters get **one final fit on the full dataset**, evaluated on a held-out split.

#### Live progress
Every slow fit reports progress: real per-iteration counts for LightGBM/CatBoost (native training callbacks), and an elapsed-time heartbeat for everything else (most sklearn estimators have no generic incremental-fit hook, so the fit runs in a background thread while a "still working — Ns elapsed" tick streams to the UI).

#### Visualisation
Two Optuna plots per model — **optimization history** and **parameter slice** — rendered via Plotly.js from `optuna.visualization`.

#### Experiment log & inference selection
- Every run (whatever models/config you try) is appended to a persistent, growing log rather than overwriting the last one — so different configurations can be compared side by side.
- Data Training's best model is seeded into the log as a **Baseline** candidate (default hyperparameters) and auto-selected for inference on first visit — this baseline stays in sync if a later Data Training run (on the same prepared dataset) finds a different best model.
- Any logged experiment can be picked via **"Use for inference"**, which persists that exact model + hyperparameters as the active choice, along with a **"Go to Inference →"** shortcut.

---

### 6. Inference

Runs the model selected on the Hyperparameter Search page against an uploaded test CSV.

#### Execution
1. Reconstructs the **exact same PyCaret preprocessing pipeline** used during training (same `session_id`, deterministic) on the full clean training data.
2. Either **loads the already-saved fitted model** (fast path — used whenever Data Training or Hyperparameter Search saved one), or **fits fresh** with the given hyperparameters (baseline/untuned path, with live progress since this can take a while for slower models).
3. Predicts on the uploaded test CSV through that same pipeline — encoding, scaling, missing values, and unseen test-time categories are all handled automatically by PyCaret's own transform, no separate manual step needed.

#### Output files
- `predictions/{id}.csv` — all input columns + `{label}_prediction` column.
- `predictions/{id}_submission.csv` — Kaggle-format submission: ID column (auto-detected or row index) + label column only.

---

## Architecture

PyCaret's dependency pins have no wheels for the Python version the main app runs on (and fail to build from source), so it lives in an **isolated second virtual environment**:

```
.venv/            Main app — FastAPI, Python 3.11+, scikit-learn/LightGBM/CatBoost (imputation)
.venv-automl/      PyCaret + Optuna + LightGBM/CatBoost, Python 3.11, provisioned from requirements-automl.txt
```

Three worker scripts run inside `.venv-automl` as **subprocesses**, launched from the FastAPI app and never imported directly — they communicate back over stdout as JSON-lines, forwarded live as Server-Sent Events:

| Worker | Used by | Does |
|---|---|---|
| `automl_worker.py` | Data Training | Model comparison + best-model save |
| `tune_worker.py` | Hyperparameter Search | Optuna search + full-dataset refit; also exports shared `build_model`/`fit_with_progress`/`run_with_heartbeat` helpers imported by the other two workers |
| `infer_worker.py` | Inference | Real prediction execution |

The FastAPI app itself is a thin `main.py` that mounts a **router per concern**:

```
routers/
├── pages.py          # static page routes
├── upload.py         # /upload, /upload/batch, /select-columns
├── analysis.py        # /analyze
├── preparation.py     # /prepare + downloads
├── training.py        # /train/* — shells out to automl_worker.py
├── hyperparameter.py  # /tune/* — shells out to tune_worker.py
├── inference.py       # /infer, /infer/pycaret/* — shells out to infer_worker.py
└── resources.py       # /health, /resources
```

`common.py` holds shared storage paths, the in-memory analysis cache, and small cross-router utilities (the SSE formatter, the id-column regex).

---

## Resource Monitoring

A live resource bar appears below the navigation on every page, polling `/resources` every 5 seconds:
- **CPU %** — measured from `/proc/stat` over a 200 ms window.
- **RAM** — used / total from `/proc/meminfo`.
- **GPU** — utilisation % and VRAM from `nvidia-smi --query-gpu` (shown only when a GPU is detected).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| AutoML / tuning | PyCaret, Optuna (isolated `.venv-automl`, Python 3.11) |
| ML | scikit-learn, LightGBM, CatBoost |
| Visualisation | Plotly.js (Optuna plots), Chart.js (exploration charts) |
| Serialisation | joblib |
| Frontend | Vanilla HTML / CSS / JS |

---

## Project Structure

```
kaggler/
├── main.py                  # FastAPI app — creates the app, mounts routers
├── common.py                 # Shared paths, analysis cache, SSE helper, id-column regex
├── automl_worker.py           # Data Training's model-comparison subprocess (.venv-automl)
├── tune_worker.py              # Hyperparameter Search's Optuna subprocess (.venv-automl)
├── infer_worker.py             # Inference's prediction subprocess (.venv-automl)
├── routers/                    # One APIRouter per concern (see Architecture)
├── requirements.txt            # Main app deps
├── requirements-automl.txt     # Frozen .venv-automl deps + provisioning instructions
├── static/
│   ├── index.html              # Data Collection
│   ├── analysis.html           # Data Exploration
│   ├── preparation.html        # Data Preparation
│   ├── training.html           # Data Training (PyCaret AutoML comparison)
│   ├── hyperparameter.html     # Hyperparameter Search (Optuna)
│   └── inference.html          # Inference & submission
├── uploads/                    # Raw uploaded CSVs (runtime, gitignored)
├── prepared/                   # Imputed / encoded CSVs + joblib bundles (gitignored)
├── models/                     # Saved model bundles (gitignored)
└── predictions/                # Prediction & submission CSVs (gitignored)
```

---

## Running the Server

**Main app:**
```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

**AutoML environment** (separate — required for Data Training, Hyperparameter Search, and Inference):
```bash
uv python install 3.11
uv venv .venv-automl --python 3.11
uv pip install --python .venv-automl/bin/python -r requirements-automl.txt
```

**Run:**
```bash
.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open `http://localhost:8000` in your browser.
