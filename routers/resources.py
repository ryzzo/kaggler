"""GPU detection and system resource endpoints (/health, /resources)."""
import os
import platform
import subprocess
import time
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()


def _detect_gpu() -> bool:
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


GPU_AVAILABLE = _detect_gpu()
print(f"[startup] GPU available: {GPU_AVAILABLE}")


def _read_ram() -> dict:
    """Return total/used/free RAM in MB and percent used."""
    info = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            info[k.strip()] = int(v.split()[0])  # kB
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        used  = total - avail
        return {
            "total_mb": round(total / 1024, 1),
            "used_mb":  round(used  / 1024, 1),
            "free_mb":  round(avail / 1024, 1),
            "percent":  round(used / total * 100, 1) if total else 0,
        }
    except Exception:
        return {}


_cpu_prev: list[int] = []


def _read_cpu() -> float:
    """Estimate CPU % over ~200 ms using /proc/stat."""
    global _cpu_prev
    def _stat():
        line = Path("/proc/stat").read_text().splitlines()[0].split()
        nums = list(map(int, line[1:]))
        idle  = nums[3]
        total = sum(nums)
        return total, idle
    try:
        t1, i1 = _stat()
        time.sleep(0.2)
        t2, i2 = _stat()
        dt, di = t2 - t1, i2 - i1
        return round((1 - di / dt) * 100, 1) if dt else 0.0
    except Exception:
        return 0.0


def _read_gpu() -> dict:
    """Return GPU utilisation and VRAM via nvidia-smi CSV query."""
    if not GPU_AVAILABLE:
        return {}
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return {}
        parts = out.stdout.strip().split(",")
        return {
            "util_pct":   float(parts[0].strip()),
            "vram_used_mb": float(parts[1].strip()),
            "vram_total_mb": float(parts[2].strip()),
            "vram_pct":   round(float(parts[1]) / float(parts[2]) * 100, 1),
        }
    except Exception:
        return {}


@router.get("/health")
def health():
    return {"status": "ok", "gpu": GPU_AVAILABLE}


@router.get("/resources")
def resources():
    return {
        "cpu_pct": _read_cpu(),
        "ram": _read_ram(),
        "gpu": _read_gpu(),
        "gpu_available": GPU_AVAILABLE,
    }


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _gpu_list() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.strip().splitlines():
            name, mem, drv = [x.strip() for x in line.split(",")]
            gpus.append({"name": name, "vram_mb": float(mem), "driver": drv})
        return gpus
    except Exception:
        return []


@lru_cache(maxsize=1)
def _torch_cuda_in_pytabkit_venv() -> dict | None:
    """Does the pytabkit venv's PyTorch see the GPU? None when that venv isn't installed."""
    from common import SCOUT_ENGINE  # noqa: F401  (import keeps config in one place)
    py = Path(".venv-pytabkit/bin/python")
    if not py.exists():
        return None
    try:
        out = subprocess.run(
            [str(py), "-c", "import torch;print(torch.cuda.is_available(), torch.version.cuda)"],
            capture_output=True, text=True, timeout=60)
        ok, _, cuda = out.stdout.strip().partition(" ")
        return {"cuda_available": ok == "True", "torch_cuda": cuda}
    except Exception:
        return {"cuda_available": False, "torch_cuda": None}


@router.get("/compute")
def compute():
    """Static hardware summary + which training engines will actually use the GPU."""
    import gpu as gpu_mod
    from common import SCOUT_ENGINE
    gpus = _gpu_list()
    cuda = gpu_mod.cuda_available()
    torch_info = _torch_cuda_in_pytabkit_venv()
    engines = [
        {"name": "CatBoost", "gpu": gpu_mod.catboost_gpu_ok()},
        {"name": "LightGBM", "gpu": gpu_mod.lightgbm_gpu_ok(),
         "note": "needs a GPU-enabled LightGBM build" if cuda and not gpu_mod.lightgbm_gpu_ok() else None},
        {"name": "PyCaret (XGBoost/others)", "gpu": cuda},
    ]
    if torch_info is not None:
        engines.append({"name": "pytabkit (PyTorch)", "gpu": cuda and torch_info["cuda_available"],
                        "note": "GPU found but PyTorch is a CPU-only build" if cuda and not torch_info["cuda_available"] else None})
    ram = _read_ram()
    return {
        "cpu": {"model": _cpu_model(),
                "logical_cores": os.cpu_count(),
                "usable_cores": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()},
        "ram_total_mb": ram.get("total_mb"),
        "gpus": gpus,
        "gpu_in_use": cuda,
        "gpu_disabled_by_env": os.environ.get("KAGGLER_USE_GPU", "auto").strip().lower() in ("0", "false", "off", "no", "cpu"),
        "scout_engine": SCOUT_ENGINE,
        "engines": engines,
    }


@router.post("/reset")
def reset():
    """Clear every working directory and in-memory cache — used when returning to
    Data Collection to start a completely new task with a clean slate. This is a
    single-user local tool (no per-session scoping), so "reset" means everything."""
    from common import MODELS_DIR, PREDICTIONS_DIR, PREPARED_DIR, UPLOAD_DIR, _analysis_cache
    from routers import hyperparameter, inference, training

    cleared_files = 0
    for d in (UPLOAD_DIR, PREPARED_DIR, MODELS_DIR, PREDICTIONS_DIR):
        for f in d.iterdir():
            if f.is_file():
                f.unlink()
                cleared_files += 1

    _analysis_cache.clear()
    training._job_store.clear()
    hyperparameter._job_store.clear()
    inference._pycaret_infer_jobs.clear()

    return {"status": "ok", "cleared_files": cleared_files}
