from pydantic import BaseModel, Field, field_validator

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYXBZJUO")


class PredictRequest(BaseModel):
    sequence: str = Field(..., min_length=1, max_length=1024)

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        v = v.strip().upper()
        invalid = set(v) - AMINO_ACIDS
        if invalid:
            raise ValueError(f"sequence contains invalid amino acid codes: {sorted(invalid)}")
        return v


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    sequence_length: int
    embedding_dim: int
    embedding: list[float]
    model_name: str


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_name: str
    device: str
