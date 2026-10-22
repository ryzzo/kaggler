"""Minimal, picklable preprocessing for the pytabkit engine.

Lives in its own importable module (not in the worker script) so a joblib bundle
saved by pytabkit_worker.py can be unpickled later by any process.

pytabkit's models take a DataFrame, detect categorical columns from dtype, and
handle scaling/encoding themselves — but they do NOT accept missing numeric
values. So this only (a) imputes and (b) pins category dtypes/levels.
"""
import numpy as np
import pandas as pd

MISSING_TOKEN = "__missing__"


class TabularPrep:
    def fit(self, X: pd.DataFrame) -> "TabularPrep":
        self.cat_cols_ = [c for c in X.columns
                          if X[c].dtype == object or str(X[c].dtype) in ("category", "bool", "string")]
        self.num_cols_ = [c for c in X.columns if c not in self.cat_cols_]
        self.medians_ = {c: (float(X[c].median()) if X[c].notna().any() else 0.0) for c in self.num_cols_}
        self.levels_ = {c: sorted(set(X[c].dropna().astype(str)) | {MISSING_TOKEN}) for c in self.cat_cols_}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in self.num_cols_:
            out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            out[c] = out[c].fillna(self.medians_[c]).astype("float32")
        for c in self.cat_cols_:
            s = out[c].astype(object).where(out[c].notna(), MISSING_TOKEN).astype(str)
            # unseen test-time levels collapse to the missing token
            s = s.where(s.isin(self.levels_[c]), MISSING_TOKEN)
            out[c] = pd.Categorical(s, categories=self.levels_[c])
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


class FittedBundleModel:
    """prep + estimator (+ label encoder for classification) behind predict()/predict_proba()."""

    def __init__(self, prep, estimator, classes=None):
        self.prep, self.estimator, self.classes = prep, estimator, classes

    def predict(self, X: pd.DataFrame):
        pred = self.estimator.predict(self.prep.transform(X))
        return self.classes[np.asarray(pred).astype(int)] if self.classes is not None else pred

    def predict_proba(self, X: pd.DataFrame):
        return self.estimator.predict_proba(self.prep.transform(X))
