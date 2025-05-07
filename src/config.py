from pydantic import BaseSettings

class Settings(BaseSettings):
    MODEL_NAME: str = "bge-m3"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    class Config:
        env_file = ".env"

settings = Settings() 