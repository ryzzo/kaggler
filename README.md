# Kaggler — End-to-End ML Pipeline

A browser-based machine learning pipeline designed for Kaggle competitions. Upload a CSV, explore it, clean it, train models, and generate a submission file — all through a dark-themed web UI backed by a FastAPI server.

---

## Pipeline Overview

```
Data Collection → Data Exploration → Data Preparation → Data Training → Inference
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
- The **fitted model, feature column list, and column means** (used as fallback for any missing feature values at inference) are serialised with `joblib` and stored alongside the prepared data. This bundle is loaded at inference time so test data receives **identical imputation**.

**Imputation params persistence**
- Saved to `prepared/{id}__imputation.joblib`.
- `imputation_params_id` is returned by `/prepare` and carried in `sessionStorage` so it is automatically supplied to `/infer`.

#### Feature encoding & scaling (applied after imputation)

| Column type | Transform |
|---|---|
| Categorical | **Label encoding** — each unique string is mapped to an integer code. The mapping is persisted and inverted at inference to restore original labels. |
| Numeric | **Standard scaling (z-score)** — `(x − μ) / σ`. Mean and std are persisted and used to inverse-transform regression predictions back to the original unit. |

**Output files**
- `prepared/{id}.csv` — encoded + scaled dataset (used for tree-based model training).
- `prepared/{id}__clean.csv` — imputed but *not* transformed dataset (original scale; selectable for training).

---

### 4. Data Training

#### Task detection
The label column's presence in `encoding_info` (categorical) or `scaling_info` (numeric) determines whether to run **classification** or **regression** variants of each model.

#### Models trained

| Model | Library | GPU support |
|---|---|---|
| Gradient Boosting | `sklearn.ensemble.GradientBoostingClassifier / Regressor` | No |
| LightGBM | `lightgbm.LGBMClassifier / LGBMRegressor` | Yes (`device='gpu'`) |
| CatBoost | `catboost.CatBoostClassifier / CatBoostRegressor` | Yes (`task_type='GPU'`) |

GPU availability is detected at startup via `nvidia-smi`; models fall back to CPU automatically.

#### ID column exclusion
Columns matching the pattern `^id$`, `^id[_-]`, or `[_-]id$` (case-insensitive) are excluded from the feature set to prevent data leakage.

#### Train / test split
- 80 / 20 random split.
- Stratified by label for classification when every class has ≥ 2 samples.

#### Evaluation metrics

| Task | Metrics |
|---|---|
| Classification | Accuracy, weighted F1 score |
| Regression | RMSE, MAE, R² |

#### Soft voting ensembles
All 2-model and 3-model combinations of the successfully trained base models are evaluated:
- **Classification** — average of `predict_proba` outputs across members; `argmax` gives the ensemble prediction.
- **Regression** — simple mean of `predict` outputs.

#### Model persistence
The **top-2 base models** (ranked by accuracy for classification, R² for regression) are serialised as joblib bundles to `models/{prepared_id}__{model_name}.joblib`. Each bundle stores the fitted model, feature column list, label metadata, and evaluation metric.

#### Streaming training progress
Results are streamed to the browser via **Server-Sent Events (SSE)**:
1. `POST /train/init` validates params and returns a `job_id`.
2. `GET /train/stream/{job_id}` opens an `EventSource`; the server emits `start → training → result` (per model) → `ensembles → saved → done` events.
3. The UI table updates live as each model finishes — no waiting for all three.

---

### 5. Inference

#### Test data preprocessing
Before prediction, the test CSV goes through the **same pipeline as training**:
1. Imputation params loaded from `{imputation_params_id}__imputation.joblib` — missing cells filled with the stored median, mode, or KNN model.
2. Categorical columns encoded with the stored label-encoding mappings (unseen values → code `-1`, counted and flagged in the UI).
3. Numeric columns scaled with the stored mean and std.

#### Prediction & inverse transform
- The best-performing saved model (highest score among the two persisted) is used by default.
- **Classification**: predicted integer codes are inverted back to original string labels via the reversed encoding map.
- **Regression**: predictions are inverse-scaled (`pred × σ + μ`) to the original unit.

#### Output files
- `predictions/{id}.csv` — all input columns + `{label}_prediction` column.
- `predictions/{id}_submission.csv` — Kaggle-format submission: ID column (auto-detected or row index) + label column only.

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
| Backend | Python 3.11+, FastAPI, Uvicorn |
| ML | scikit-learn, LightGBM, CatBoost |
| Serialisation | joblib |
| Frontend | Vanilla HTML / CSS / JS, Chart.js (exploration charts) |
| Data handling | Pure Python `csv` module — no pandas dependency |

---

## Project Structure

```
kaggler/
├── main.py               # FastAPI app — all routes, ML logic, SSE streaming
├── requirements.txt
├── static/
│   ├── index.html        # Data Collection
│   ├── analysis.html     # Data Exploration
│   ├── preparation.html  # Data Preparation
│   ├── training.html     # Data Training
│   └── inference.html    # Inference & submission
├── uploads/              # Raw uploaded CSVs (runtime, gitignored)
├── prepared/             # Imputed / encoded CSVs + joblib bundles (gitignored)
├── models/               # Saved model bundles (gitignored)
└── predictions/          # Prediction & submission CSVs (gitignored)
```

---

## Running the Server

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open `http://localhost:8000` in your browser.
