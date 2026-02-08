from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "DeepGuard"
    ENV: str = "development"
    DEBUG: bool = True

    class Config:
        env_file = ".env"

@lru_cache
def get_settings():
    return Settings()
