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
    #granite_picture_description - 픽처 디스크립션 모드
)

# config에서 설정 및 로거 가져오기
from src.config import logger, settings

# --- 프라이머리 어댑터 설정 함수 임포트 ---
from src.adapters.primary.api_adapter import setup_api_routes

# --- 애플리케이션 계층 유스케이스 임포트 ---
from src.application.use_cases import IngestDocumentUseCase

# --- 세컨더리 어댑터 구현체 임포트 ---
from src.adapters.secondary.docling_parser_adapter import DoclingParserAdapter
from src.adapters.secondary.docling_chunker_adapter import DoclingChunkerAdapter
from src.adapters.secondary.bge_m3_embedder_adapter import BgeM3EmbedderAdapter
#from src.adapters.secondary.env_apikey_adapter import EnvApiKeyAdapter
from src.adapters.secondary.milvus_adapter import MilvusAdapter, _milvus_library_available

# --- 더미 어댑터 클래스 정의 ---
# DummyMilvusAdapter 클래스: Milvus 연결 실패 시 사용하는 백업 어댑터
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
        
    def save_document_data(self, chunks, embeddings):
        logger.warning("DummyMilvusAdapter: save_document_data called, but no action taken")
        return {"status": "dummy", "message": "Data not persisted in vector database"}

# --- 하드코딩된 설정값 제거하고 settings에서 값을 가져오도록 수정 ---

def create_app() -> FastAPI:
    """
    FastAPI 애플리케이션 인스턴스를 생성하고,
    시스템의 모든 구성 요소를 조립(Wiring)하여 의존성을 주입합니다.
    이 함수가 헥사고날 아키텍처의 컴포지션 루트 역할을 합니다.
    """
    logger.info("--- Starting Application Assembly ---")

    # API 키 관리 어댑터
    #apikey_adapter = EnvApiKeyAdapter()
    #logger.info("- Created EnvApiKeyAdapter instance.")

    # OCR 옵션 및 파이프라인 옵션 제거, 테이블/텍스트 추출만 활성화 (이미지 설명 완전 비활성화)
    pdf_options = PdfPipelineOptions(
        do_ocr=settings.DO_OCR,  # OCR 설정
        do_table_structure=settings.DO_TABLE_STRUCTURE,  # 테이블 구조 추출 설정
        images_scale=settings.IMAGES_SCALE,  # 이미지 크기 조정
        generate_picture_images=settings.GENERATE_PICTURE_IMAGES  # 이미지 생성 설정
    )
    
    # 파이프라인 옵션 로깅 (이미지 설명 관련 부분 제거)
    logger.info("파이프라인 옵션: OCR/테이블/텍스트 추출만 활성화, 이미지 설명 완전 비활성화")
    
    # 파서 어댑터 생성
    try:
        parser_adapter = DoclingParserAdapter(
            allowed_formats=settings.DOCLING_ALLOWED_FORMATS.split(','),
            use_gpt_picture_description=False  # OpenAI 모델 사용하지 않음
        )
        # DoclingParserAdapter 내부 파이프라인 옵션도 명시적으로 비활성화
        if hasattr(parser_adapter, '_converter') and parser_adapter._converter is not None:
            pdf_format_option = parser_adapter._converter.format_to_options.get('pdf')
            if pdf_format_option and hasattr(pdf_format_option, 'pipeline_options'):
                pdf_format_option.pipeline_options.do_picture_description = False
        logger.info("- Created DoclingParserAdapter instance.")
    except Exception as e:
        logger.error(f"Failed to initialize DoclingParserAdapter: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Application startup failed: {e}")

    # 청킹 어댑터 생성
    try:
        chunker_adapter = DoclingChunkerAdapter(
            chunk_size=settings.CHUNK_SIZE, 
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        logger.info("- Created DoclingChunkerAdapter instance.")
    except Exception as e:
        logger.error(f"Failed to initialize DoclingChunkerAdapter: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Application startup failed: {e}")

    # 임베딩 어댑터 생성
    try:
        embedder_adapter = BgeM3EmbedderAdapter(
            model_name=settings.EMBEDDING_MODEL_NAME,
            device=settings.EMBEDDING_DEVICE
        )
        logger.info("- Created BgeM3EmbedderAdapter instance.")
    except Exception as e:
        logger.error(f"Failed to initialize BgeM3EmbedderAdapter: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Application startup failed: {e}")

    # Milvus 어댑터 생성 시도
    persistence_adapter = None
    try:
        # Milvus 라이브러리 사용 가능 여부 확인
        if not _milvus_library_available:
            logger.warning("pymilvus 라이브러리를 사용할 수 없습니다. 더미 어댑터를 사용합니다.")
            persistence_adapter = DummyMilvusAdapter()
            logger.info("- Created DummyMilvusAdapter (pymilvus library not available).")
        else:
            # 인증 정보 구성
            token = None
            if settings.MILVUS_USER and settings.MILVUS_PASSWORD:
                token = f"{settings.MILVUS_USER}:{settings.MILVUS_PASSWORD}"
                
            # MilvusAdapter 인스턴스 생성 시도
            persistence_adapter = MilvusAdapter(
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
                collection_name=settings.MILVUS_COLLECTION,
                token=token
            )
            logger.info("- Created MilvusAdapter instance and successfully connected.")
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
        #    api_key_port=apikey_adapter
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
        host=settings.API_HOST, 
        port=settings.API_PORT, 
        use_colors=True,
        limit_max_requests=settings.API_LIMIT_MAX_REQUESTS,  # 메모리 누수 방지를 위한 최대 요청 수 제한
        reload=settings.API_RELOAD  # 혹은 reload_dirs를 직접 지정하여 감시할 디렉토리 제한
    )

