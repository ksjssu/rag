# src/main.py -> main.py

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
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager

# src 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent / "src"))

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    granite_picture_description
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
from src.adapters.secondary.milvus_adapter import MilvusAdapter


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 시작/종료 시 실행되는 이벤트 핸들러
    """
    # 시작 시점
    logger.info("--- Application Startup ---")
    try:
        # 여기에 시작 시 필요한 초기화 코드
        yield
    finally:
        # 종료 시점
        logger.info("--- Application Shutdown ---")
        # 여기에 종료 시 필요한 정리 코드

def create_app() -> FastAPI:
    """
    FastAPI 애플리케이션 인스턴스를 생성하고,
    시스템의 모든 구성 요소를 조립(Wiring)하여 의존성을 주입합니다.
    이 함수가 헥사고날 아키텍처의 컴포지션 루트 역할을 합니다.
    """
    logger.info("--- Starting Application Assembly ---")

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

    # Milvus 어댑터 생성
    try:
        # 인증 정보 구성
        token = None
        if settings.MILVUS_USER and settings.MILVUS_PASSWORD:
            token = f"{settings.MILVUS_USER}:{settings.MILVUS_PASSWORD}"
            
        # MilvusAdapter 인스턴스 생성
        persistence_adapter = MilvusAdapter(
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            collection_name=settings.MILVUS_COLLECTION,
            token=token
        )
        logger.info("- Created MilvusAdapter instance and successfully connected.")
    except Exception as e:
        logger.error(f"Failed to initialize MilvusAdapter: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Application startup failed: {e}")

    # 유스케이스 인스턴스 생성
    try:
        ingest_use_case_instance = IngestDocumentUseCase(
            parser_port=parser_adapter,
            chunking_port=chunker_adapter,
            embedding_port=embedder_adapter,
            persistence_port=persistence_adapter,
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
        lifespan=lifespan  # lifespan 추가
    )
    logger.info("- Created FastAPI application instance.")

    # Prometheus Instrumentator 설정
    instrumentator = Instrumentator()
    instrumentator.instrument(app)
    app.instrumentator = instrumentator
    logger.info("- Configured Prometheus Instrumentator.")

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


