FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime deps only (no torch/analysis stack). onnxruntime enables /api/classify when a model is mounted.
COPY requirements.txt requirements-serve.txt ./
RUN pip install -r requirements.txt -r requirements-serve.txt gunicorn==23.0.0

COPY . .

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/data', timeout=4)"

# Threads matter: /video_feed holds a connection open per viewer.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "8", "--timeout", "60", "app:app"]
