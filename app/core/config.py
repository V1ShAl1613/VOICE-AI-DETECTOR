import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_KEY: str = os.getenv("API_KEY", "secret123")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "app/ml/voice_auth_model.pkl")
    # Threshold for AI classification (>= threshold means AI)
    AI_PROBABILITY_THRESHOLD: float = float(os.getenv("AI_PROBABILITY_THRESHOLD", "0.6"))
    
    class Config:
        env_file = ".env"

settings = Settings()
