"""GPU helpers shared by the worker scripts (tune_worker.py, automl_worker.py).

Stdlib-only at import time so it is safe to import from either venv.

Behaviour is controlled by the KAGGLER_USE_GPU environment variable:
  auto (default) - use the GPU when an NVIDIA GPU is detected and the library
                   actually works with it (verified by a tiny probe fit);
                   otherwise silently fall back to CPU
  1 / true / on  - same as auto, but treat a missing GPU as unavailable anyway
                   (probes still protect against broken installs)
  0 / false / off / cpu - never use the GPU
"""
import os
import subprocess
from functools import lru_cache

_OFF = {"0", "false", "off", "no", "cpu"}


def _mode() -> str:
    return os.environ.get("KAGGLER_USE_GPU", "auto").strip().lower()


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    """True when GPU use is not disabled and nvidia-smi reports a GPU."""
    if _mode() in _OFF:
        return False
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=5)
        return result.returncode == 0 and b"GPU" in result.stdout
    except (OSError, subprocess.SubprocessError):
        return False


@lru_cache(maxsize=1)
def catboost_gpu_ok() -> bool:
    """CatBoost GPU training works (CUDA wheels ship with the pip package)."""
    if not cuda_available():
        return False
    try:
        import numpy as np
        from catboost import CatBoostClassifier

        X = np.random.rand(64, 4)
        y = np.arange(64) % 2
        CatBoostClassifier(iterations=2, task_type="GPU", verbose=False,
                           allow_writing_files=False).fit(X, y)
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def lightgbm_gpu_ok() -> bool:
    """LightGBM was built with GPU support (the default pip wheel often is not)."""
    if not cuda_available():
        return False
    try:
        import numpy as np
        from lightgbm import LGBMClassifier

        X = np.random.rand(64, 4)
        y = np.arange(64) % 2
        LGBMClassifier(n_estimators=2, device_type="gpu", verbose=-1,
                       min_child_samples=1).fit(X, y)
        return True
    except Exception:
        return False


def gpu_kwargs(model_id: str) -> dict:
    """Extra estimator kwargs that move `model_id` onto the GPU, or {} if it
    has no usable GPU mode here."""
    if model_id == "catboost" and catboost_gpu_ok():
        return {"task_type": "GPU"}
    if model_id == "lightgbm" and lightgbm_gpu_ok():
        return {"device_type": "gpu"}
    return {}


def pycaret_use_gpu() -> bool:
    """Value for PyCaret's setup(use_gpu=...). PyCaret itself falls back to CPU
    for any model whose GPU backend (cuML, xgboost-gpu, ...) isn't installed."""
    return cuda_available()


def describe() -> dict:
    return {
        "cuda": cuda_available(),
        "catboost_gpu": catboost_gpu_ok(),
        "lightgbm_gpu": lightgbm_gpu_ok(),
    }
