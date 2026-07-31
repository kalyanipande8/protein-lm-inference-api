import pytest
from fastapi.testclient import TestClient

from app import model as model_module
from app.main import app


class FakeModel:
    """Deterministic stand-in for ProteinEmbeddingModel so tests don't
    require downloading real ESM-2 weights."""

    model_name = "fake-esm2"
    device = "cpu"
    embedding_dim = 4

    def embed_batch(self, sequences: list[str]) -> list[list[float]]:
        return [[float(len(seq))] * self.embedding_dim for seq in sequences]


@pytest.fixture
def client():
    model_module.set_model(FakeModel())
    with TestClient(app) as c:
        yield c
    model_module.set_model(None)
