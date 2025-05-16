from pydantic_settings import BaseSettings
import logging
from typing import Optional, List

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
    # 임베딩 모델 기본 설정
    MODEL_NAME: str = "BAAI/bge-m3"
    
    # 청킹 설정 - 중복 변수 제거하고 하나로 통일
    CHUNK_SIZE: int = 3000      # 기본 청크 크기 (문자 단위)
    CHUNK_OVERLAP: int = 300    # 청크 간 오버랩 크기
    
    # Milvus 설정
    MILVUS_HOST: str = "10.10.30.80"
    MILVUS_PORT: int = 30953
    MILVUS_COLLECTION: str = "colbert_test"
    MILVUS_USER: Optional[str] = "root"
    MILVUS_PASSWORD: Optional[str] = "smr0701!"
    
    # Docling 설정
    DOCLING_ALLOWED_FORMATS: str = "pdf,docx,xlsx,pptx,jpg,png"
    
    # 임베딩 모델 설정
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"       # GPU 모델 사용 시 "cuda:0" 
    
    # Docling 파이프라인 설정
    DO_OCR: bool = False
    DO_TABLE_STRUCTURE: bool = True
    IMAGES_SCALE: float = 2.0
    GENERATE_PICTURE_IMAGES: bool = True
    
    # API 서버 설정
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_LIMIT_MAX_REQUESTS: int = 100
    API_RELOAD: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings() 