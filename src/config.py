from pydantic_settings import BaseSettings
import logging

# 로깅 설정
def setup_logger():
    logger = logging.getLogger("rag")
    # 핸들러가 이미 있는지 확인하고 중복 추가 방지
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    # 상위 로거로 전파 방지 (중복 로그 방지)
    logger.propagate = False
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