import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    model_path: str
    vocab_path: str
    max_len: int = 10000
    allowed_origins: List[str] = ["*"]
    google_api_key: str = ""

    class Config:
        env_file = os.path.join(os.path.dirname(__file__), "../../.env")
        env_file_encoding = 'utf-8'

settings = Settings()