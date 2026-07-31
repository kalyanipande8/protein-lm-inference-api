# Protein LM Inference API

A small FastAPI service that serves sequence embeddings from a pretrained
ESM-2 protein language model ([`facebook/esm2_t6_8M_UR50D`](https://huggingface.co/facebook/esm2_t6_8M_UR50D)),
with server-side dynamic batching and a Dockerized deployment.

## What this actually does

- Loads ESM-2 via Hugging Face `transformers` and exposes it over a REST API.
- `POST /predict` accepts a single amino acid sequence and returns a
  mean-pooled embedding vector (special tokens excluded from the pool).
- Concurrent requests are collected server-side by a `DynamicBatcher`
  ([app/batcher.py](app/batcher.py)) that waits up to `PLM_MAX_BATCH_WAIT_MS`
  (default 10ms) or until `PLM_MAX_BATCH_SIZE` (default 16) requests have
  queued, then runs one forward pass for the whole batch — so throughput
  under concurrent load doesn't scale linearly with per-request inference
  cost.
- Containerized with a Dockerfile that pre-downloads model weights at build
  time.

This is a base model wrapped for serving, not a fine-tuned model — there is
no training script or labeled dataset in this repo. If you want to back the
"fine-tuned on N sequences" framing, add a fine-tuning script against a real
dataset (e.g. a UniProt subset) before using that language anywhere.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then:

```bash
curl -X POST localhost:8000/predict \
  -H "content-type: application/json" \
  -d '{"sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWELVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"}'
curl localhost:8000/health
```

## Docker

```bash
docker build -t protein-lm-inference-api .
docker run -p 8000:8000 protein-lm-inference-api
```

## Tests

```bash
pytest
```

Tests inject a fake model (see [tests/conftest.py](tests/conftest.py)) so
they run fast and offline — they exercise request validation, the
`/predict`/`/health` contract, and that concurrent requests all get correct
per-sequence results back from the shared batcher.

## Measuring latency under load

[scripts/load_test.py](scripts/load_test.py) fires concurrent requests at a
running instance and reports p50/p95/p99 latency:

```bash
uvicorn app.main:app &
python scripts/load_test.py --url http://localhost:8000 --requests 500 --concurrency 50
```

Actual p95 will depend on hardware, sequence length distribution, and the
batching parameters above — measure it on your target deployment rather than
assuming a number.

## Configuration

Environment variables (prefix `PLM_`):

| Variable | Default | Description |
|---|---|---|
| `PLM_MODEL_NAME` | `facebook/esm2_t6_8M_UR50D` | HF model id |
| `PLM_DEVICE` | `cpu` | `cpu` or `cuda` |
| `PLM_MAX_BATCH_SIZE` | `16` | Max sequences per forward pass |
| `PLM_MAX_BATCH_WAIT_MS` | `10` | Max wait to fill a batch |
| `PLM_MAX_SEQUENCE_LENGTH` | `1024` | Truncation length |

## Not included yet

- Fine-tuning script / labeled dataset
- Infrastructure-as-code for cloud deployment (Terraform, ECS/Cloud Run, etc.)
- GPU-specific batching tuning
