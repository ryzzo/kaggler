# syntax=docker/dockerfile:1
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

# libgomp1: OpenMP runtime required by LightGBM and CatBoost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Same two-venv layout as local dev (see README "Architecture"): common.py resolves
# .venv-automl/bin/python relative to the working directory, so both live in /app.
# The AutoML env is the heavy, rarely-changing one — build it first for layer caching.
COPY requirements-automl.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv .venv-automl --python /usr/local/bin/python3.11 \
    && uv pip install --python .venv-automl/bin/python -r requirements-automl.txt

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv .venv --python /usr/local/bin/python3.11 \
    && uv pip install --python .venv/bin/python -r requirements.txt

COPY . .

# Run as non-root. /app itself must be writable too — CatBoost writes catboost_info/
# into the working directory.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p uploads prepared models predictions \
    && chown app:app /app uploads prepared models predictions
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD .venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD [".venv/bin/python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
