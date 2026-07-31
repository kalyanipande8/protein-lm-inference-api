import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.batcher import DynamicBatcher
from app.config import settings
from app.model import get_model
from app.schemas import HealthResponse, PredictRequest, PredictResponse

logger = logging.getLogger("protein_lm_api")

batcher: DynamicBatcher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global batcher
    model = get_model()
    batcher = DynamicBatcher(model)
    batcher.start()
    logger.info("Loaded model %s on %s", settings.model_name, settings.device)
    yield
    await batcher.stop()


app = FastAPI(title="Protein LM Inference API", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    model = get_model()
    return HealthResponse(status="ok", model_name=model.model_name, device=model.device)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    if batcher is None:
        raise HTTPException(status_code=503, detail="model not ready")

    try:
        embedding = await batcher.predict(request.sequence)
    except Exception as exc:  # noqa: BLE001
        logger.exception("inference failed")
        raise HTTPException(status_code=500, detail="inference failed") from exc

    return PredictResponse(
        sequence_length=len(request.sequence),
        embedding_dim=len(embedding),
        embedding=embedding,
        model_name=get_model().model_name,
    )
