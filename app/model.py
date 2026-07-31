import torch
from transformers import AutoModel, AutoTokenizer

from app.config import settings


class ProteinEmbeddingModel:
    """Wraps an ESM-2 encoder and produces mean-pooled sequence embeddings."""

    def __init__(self, model_name: str = settings.model_name, device: str = settings.device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size

    @torch.inference_mode()
    def embed_batch(self, sequences: list[str]) -> list[list[float]]:
        """Runs a single forward pass over a batch of sequences and returns
        one mean-pooled embedding per sequence (special tokens excluded)."""
        encoded = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=settings.max_sequence_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoded)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # exclude padding + special tokens (BOS/EOS) from the mean-pool
        mask = encoded["attention_mask"].clone()
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            encoded["input_ids"][0].tolist(), already_has_special_tokens=True
        )
        # per-sequence special-token exclusion (BOS/EOS positions differ only
        # by padding, so recompute mask per row using each row's token ids)
        for i, ids in enumerate(encoded["input_ids"].tolist()):
            row_special = self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
            for j, is_special in enumerate(row_special):
                if is_special:
                    mask[i, j] = 0

        mask = mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts

        return pooled.cpu().tolist()


_model: ProteinEmbeddingModel | None = None


def get_model() -> ProteinEmbeddingModel:
    global _model
    if _model is None:
        _model = ProteinEmbeddingModel()
    return _model


def set_model(model: ProteinEmbeddingModel) -> None:
    """Test hook to inject a fake/mock model."""
    global _model
    _model = model
