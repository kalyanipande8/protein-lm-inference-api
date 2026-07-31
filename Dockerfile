FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Pre-download model weights at build time so the container doesn't hit
# HuggingFace Hub on first request.
ENV PLM_MODEL_NAME=facebook/esm2_t6_8M_UR50D
RUN python -c "from app.model import ProteinEmbeddingModel; ProteinEmbeddingModel()"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
