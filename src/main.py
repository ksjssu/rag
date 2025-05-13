# src/main.py

# .env 파일에서 환경 변수를 로드합니다.
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
import uvicorn
import os
import logging
from typing import Optional, List
from pathlib import Path
import sys
import traceback

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    granite_picture_description
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 프라이머리 어댑터 설정 함수 임포트 ---
from src.adapters.primary.api_adapter import setup_api_routes

# --- 애플리케이션 계층 유스케이스 임포트 ---
from src.application.use_cases import IngestDocumentUseCase

# --- 세컨더리 어댑터 구현체 임포트 ---
from src.adapters.secondary.docling_parser_adapter import DoclingParserAdapter
from src.adapters.secondary.docling_chunker_adapter import DoclingChunkerAdapter
from src.adapters.secondary.bge_m3_embedder_adapter import BgeM3EmbedderAdapter
from src.adapters.secondary.env_apikey_adapter import EnvApiKeyAdapter
from src.adapters.secondary.milvus_adapter import MilvusAdapter

# --- 애플리케이션 설정 로드 ---
# 환경 변수에서 설정을 로드하거나 기본값을 사용합니다.
MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", 19530))
MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "document_collection")
MILVUS_USER: Optional[str] = os.getenv("MILVUS_USER", "")
MILVUS_PASSWORD: Optional[str] = os.getenv("MILVUS_PASSWORD", "")

# Docling 설정
DOCLING_ALLOWED_FORMATS: List[str] = os.getenv("DOCLING_ALLOWED_FORMATS", "pdf,docx,xlsx,pptx,jpg,png").split(',')
DOCLING_OCR_LANGUAGES: List[str] = os.getenv("DOCLING_OCR_LANGUAGES", "kor,eng").split(',')

# 청킹 설정
DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", 1000))
DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 200))

# 임베딩 모델 설정
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")

# 더미 Milvus 어댑터 클래스 정의 (실제 Milvus를 사용할 수 없을 때 대체용)
class DummyMilvusAdapter:
    """Milvus 연결 실패 시 사용하는 더미 어댑터"""
    
    def __init__(self, *args, **kwargs):
        logger.warning("Using DummyMilvusAdapter - Data will not be persisted")
        
    async def store_document(self, *args, **kwargs):
        logger.warning("DummyMilvusAdapter: store_document called, but no action taken")
        return {"status": "dummy", "message": "Data not stored (using dummy adapter)"}

    async def search_documents(self, *args, **kwargs):
        logger.warning("DummyMilvusAdapter: search_documents called, but no action taken")
        return []

    async def get_document(self, *args, **kwargs):
        logger.warning("DummyMilvusAdapter: get_document called, but no action taken")
        return None


def create_app() -> FastAPI:
    """
    FastAPI 애플리케이션 인스턴스를 생성하고,
    시스템의 모든 구성 요소를 조립(Wiring)하여 의존성을 주입합니다.
    이 함수가 헥사고날 아키텍처의 컴포지션 루트 역할을 합니다.
    """
    logger.info("--- Starting Application Assembly ---")

    # API 키 관리 어댑터
    apikey_adapter = EnvApiKeyAdapter()
    logger.info("- Created EnvApiKeyAdapter instance.")

    # OCR 옵션 설정
    ocr_options = EasyOcrOptions(
        lang=["ko", "en"],  # 한국어와 영어 지원
        confidence_threshold=0.3,
        download_enabled=True
    )
    ocr_options.force_full_page_ocr = True

    # 기본 파이프라인 옵션 설정
    pdf_options = PdfPipelineOptions(
        do_ocr=True,  # OCR 활성화
        ocr_options=ocr_options,  # OCR 옵션 설정
        generate_page_images=True,  # OCR을 위한 이미지 생성
        do_picture_description=True,  # 이미지 설명 활성화
        picture_description_options=granite_picture_description,  # 기본 내장 모델 사용
        images_scale=2.0,  # 이미지 크기 조정
        generate_picture_images=True  # 이미지 생성 활성화
    )
    
    # 파이프라인 옵션 로깅
    logger.info(f"Type of pdf_options.picture_description_options: {type(pdf_options.picture_description_options)}")
    logger.info(f"Value of pdf_options.picture_description_options: {pdf_options.picture_description_options!r}")

    # granite_picture_description이 객체이고 prompt 속성을 가지고 있는지 확인
    if hasattr(pdf_options.picture_description_options, 'prompt'):
        logger.info(f"Prompt on pdf_options.picture_description_options: {pdf_options.picture_description_options.prompt!r}")
    else:
        logger.info("pdf_options.picture_description_options does not have a 'prompt' attribute directly, or it might have been overwritten.")
            
    # 이미지 설명 옵션 활성화
    logger.info("이미지 설명 기능이 활성화되었습니다.")
    
    # 파서 어댑터 생성
    try:
        parser_adapter = DoclingParserAdapter(
            allowed_formats=DOCLING_ALLOWED_FORMATS,
            use_gpt_picture_description=False  # OpenAI 모델 사용하지 않음
        )
        logger.info("- Created DoclingParserAdapter instance.")
    except Exception as e:
        logger.error(f"Failed to initialize DoclingParserAdapter: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Application startup failed: {e}")

    # 청킹 어댑터 생성
    try:
        chunker_adapter = DoclingChunkerAdapter(
            chunk_size=DEFAULT_CHUNK_SIZE, 
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        logger.info("- Created DoclingChunkerAdapter instance.")
    except Exception as e:
        logger.error(f"Failed to initialize DoclingChunkerAdapter: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Application startup failed: {e}")

    # 임베딩 어댑터 생성
    try:
        embedder_adapter = BgeM3EmbedderAdapter(
            model_name=EMBEDDING_MODEL_NAME,
            device=EMBEDDING_DEVICE
        )
        logger.info("- Created BgeM3EmbedderAdapter instance.")
    except Exception as e:
        logger.error(f"Failed to initialize BgeM3EmbedderAdapter: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Application startup failed: {e}")

    # Milvus 어댑터 생성 시도
    persistence_adapter = None
    try:
        token = None
        if MILVUS_USER and MILVUS_PASSWORD:
            token = f"{MILVUS_USER}:{MILVUS_PASSWORD}"
            
        persistence_adapter = MilvusAdapter(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            collection_name=MILVUS_COLLECTION,
            token=token
        )
        logger.info("- Created MilvusAdapter instance and attempted connection.")
    except Exception as e:
        logger.warning(f"--- WARNING: Failed to initialize MilvusAdapter. Using dummy adapter instead. --- Error: {e}")
        # 실패 시 더미 어댑터 사용
        persistence_adapter = DummyMilvusAdapter()
        logger.info("- Created DummyMilvusAdapter as fallback.")

    # 유스케이스 인스턴스 생성
    try:
        ingest_use_case_instance = IngestDocumentUseCase(
            parser_port=parser_adapter,
            chunking_port=chunker_adapter,
            embedding_port=embedder_adapter,
            persistence_port=persistence_adapter,
            api_key_port=apikey_adapter
        )
        logger.info("- Created IngestDocumentUseCase instance and injected dependencies.")
    except Exception as e:
        logger.error(f"Failed to initialize IngestDocumentUseCase: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Application startup failed: {e}")

    # FastAPI 애플리케이션 인스턴스 생성
    app = FastAPI(
        title="RAG Hexagonal Document Ingestion System",
        description="Document ingestion and processing system based on Hexagonal Architecture, using Docling and Milvus.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    logger.info("- Created FastAPI application instance.")

    # API 라우터 설정
    api_router = setup_api_routes(input_port=ingest_use_case_instance)
    logger.info("- Configured API router with IngestDocumentUseCase.")
    app.include_router(api_router)
    logger.info("- Included API router in FastAPI app.")

    # 미들웨어 설정
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """HTTP 요청 및 응답을 로깅하는 미들웨어"""
        logger.info(f"요청 수신: {request.url.path}")
        try:
            response = await call_next(request)
            logger.info(f"응답 송신: {response.status_code}")
            return response
        except Exception as e:
            logger.error(f"오류 발생: {str(e)}")
            raise

    # 기본 라우트 추가
    @app.get("/")
    async def root():
        return {"message": "RAG Document Processing API is running. See /docs for API documentation."}

    @app.get("/health")
    async def health_check():
        """간단한 헬스 체크 엔드포인트"""
        return {"status": "healthy"}

    logger.info("--- Application Assembly Complete ---")
    logger.info("Application is ready.")
    logger.info(f"__name__ 값: {__name__}")

    # 조립이 완료된 FastAPI 애플리케이션 인스턴스 반환
    return app

# 전역 app 변수 생성
app = None

# 초기 시작 시도
try:
    app = create_app()
    logger.info(f"애플리케이션 조립 완료: {app}")
except Exception as e:
    logger.critical(f"--- FATAL ERROR: Application startup failed --- Error: {e}")
    # 오류가 있더라도 최소한의 앱 생성 (상태 체크용)
    app = FastAPI(title="RAG API (Error State)")
    
    @app.get("/")
    async def error_root():
        return {"status": "error", "message": f"Application startup failed: {str(e)}"}
    
    @app.get("/health")
    async def error_health():
        return {"status": "unhealthy", "message": str(e)}

# 개발 모드에서 직접 실행 시
if __name__ == "__main__":
    # 파일 변경 감시 설정을 조정하여 메모리 오류 방지
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        use_colors=True,
        limit_max_requests=100,  # 메모리 누수 방지를 위한 최대 요청 수 제한
        reload=False  # 혹은 reload_dirs를 직접 지정하여 감시할 디렉토리 제한
    )

