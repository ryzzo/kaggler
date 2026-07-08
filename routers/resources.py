"""GPU detection and system resource endpoints (/health, /resources)."""
import subprocess
import time
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
