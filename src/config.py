from pydantic_settings import BaseSettings
import logging

# 로깅 설정
def setup_logger():
    logger = logging.getLogger("rag")
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

# 애플리케이션 로거 생성
logger = setup_logger()

class Settings(BaseSettings):
    MODEL_NAME: str = "bge-m3"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings() 