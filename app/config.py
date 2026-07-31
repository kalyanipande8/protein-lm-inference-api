from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PLM_", protected_namespaces=())

    model_name: str = "facebook/esm2_t6_8M_UR50D"
    device: str = "cpu"

    # dynamic micro-batching
    max_batch_size: int = 16
    max_batch_wait_ms: int = 10
    max_sequence_length: int = 1024


settings = Settings()
