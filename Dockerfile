# Backend API container for Cloud Run.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    PYTHONPATH=/app \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# libgomp1 is required by LightGBM/scikit-learn wheels at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY prod-requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r prod-requirements.txt

RUN useradd --create-home --shell /usr/sbin/nologin app

COPY --chown=app:app src ./src
COPY --chown=app:app data ./data

RUN mkdir -p /app/data/db /app/data/processed /tmp/matplotlib \
    && if [ -f /app/data/serving/movies.db ]; then cp /app/data/serving/movies.db /app/data/db/movies.db; fi \
    && chown -R app:app /app /tmp/matplotlib

EXPOSE 8080

USER app

CMD ["sh", "-c", "python -m uvicorn src.app.main:app --host 0.0.0.0 --port ${PORT}"]
