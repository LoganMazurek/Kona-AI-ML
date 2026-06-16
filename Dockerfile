FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Cap glibc per-thread malloc arenas to curb RSS bloat/fragmentation in the
    # threaded gunicorn worker (default is 8*nproc arenas, which balloons idle RSS).
    MALLOC_ARENA_MAX=2 \
    # Single-event inference needs no intra-op parallelism. Pinning the OpenMP /
    # BLAS thread pools to 1 stops each ML lib (xgboost/catboost/lightgbm) and
    # numpy from reserving per-core thread stacks/buffers, cutting idle memory and
    # avoiding CPU oversubscription against the co-located Neighborhood Route app.
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "2", "--timeout", "120", "--max-requests", "500", "--max-requests-jitter", "50", "wsgi:app"]
