FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PORT=8000

WORKDIR /app

# libgomp1 is commonly needed by scikit-learn/numpy wheels at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY src ./src
COPY artifacts ./artifacts
COPY data ./data
COPY scripts ./scripts

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.app.main:app --host 0.0.0.0 --port ${PORT}"]