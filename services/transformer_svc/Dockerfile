# syntax=docker/dockerfile:1.6
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# (git seulement si nécessaire; sinon, supprime la ligne suivante)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY main.py ./main.py
EXPOSE 8002
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8002"]
