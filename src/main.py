# src/main.py

# .env 파일에서 환경 변수를 로드합니다.
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
import uvicorn
import os
from typing import Optional, List

# --- 프라이머리 어댑터 설정 함수 임포트 ---
# 기존: from adapters.primary.api_adapter import setup_api_routes
from .adapters.primary.api_adapter import setup_api_routes # <-- 상대 경로 임포트로 변경 (.adapters...)

# --- 애플리케이션 계층 유스케이스 임포트 ---
# 기존: from application.use_cases import IngestDocumentUseCase
from .application.use_cases import IngestDocumentUseCase # <-- 상대 경로 임포트로 변경 (.application...)

# --- 세컨더리 어댑터 구현체 임포트 ---
# 기존: from adapters.secondary.docling_parser_adapter import DoclingParserAdapter
# 기존: from adapters.secondary.docling_chunker_adapter import DoclingChunkerAdapter
# 기존: from adapters.secondary.bge_m3_embedder_adapter import BgeM3EmbedderAdapter
# 기존: from adapters.secondary.env_apikey_adapter import EnvApiKeyAdapter
# 기존: from adapters.secondary.milvus_adapter import MilvusAdapter

from .adapters.secondary.docling_parser_adapter import DoclingParserAdapter # <-- 상대 경로 임포트로 변경 (.adapters...)
from .adapters.secondary.docling_chunker_adapter import DoclingChunkerAdapter # <-- 상대 경로 임포트로 변경 (.adapters...)
from .adapters.secondary.bge_m3_embedder_adapter import BgeM3EmbedderAdapter # <-- 상대 경로 임포트로 변경 (.adapters...)
from .adapters.secondary.env_apikey_adapter import EnvApiKeyAdapter # <-- 상대 경로 임포트로 변경 (.adapters...)
from .adapters.secondary.milvus_adapter import MilvusAdapter # <-- 상대 경로 임포트로 변경 (.adapters...)


# --- 애플리케이션 설정 로드 ---
# 실제 애플리케이션에서는 환경 변수, 설정 파일 등에서 로드합니다.
# 여기서는 환경 변수에서 로드하거나 기본값을 사용하는 예시를 보여줍니다.
# from config import load_config # 예시: 별도 설정 로딩 함수
# app_config = load_config() # 예시

# Milvus 연결 설정 (예시 값 - 실제 환경에 맞게 수정 필요)
# 환경 변수에서 값을 읽어옵니다. 환경 변수가 설정되지 않았다면 기본값을 사용합니다.
MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", 19530))
MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "rag_chunks")
MILVUS_USER: Optional[str] = os.getenv("MILVUS_USER", None) # 환경 변수 없으면 None
MILVUS_PASSWORD: Optional[str] = os.getenv("MILVUS_PASSWORD", None) # 환경 변수 없으면 None

# Docling 설정 (예시)
DOCLING_ALLOWED_FORMATS: List[str] = os.getenv("DOCLING_ALLOWED_FORMATS", "pdf,docx,xlsx,pptx,jpg,png").split(',') # 쉼표로 구분된 문자열을 리스트로 변환
# DOCLING_API_KEY = os.getenv("DOCLING_API_KEY") # EnvApiKeyAdapter가 처리 가능

# 청킹 설정 (심플 청킹 또는 Docling HybridChunker에 전달될 설정)
DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", 1000))
DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 200))
# HybridChunker 특정 설정도 여기에 포함될 수 있습니다.
# HYBRID_CHUNKER_STRATEGIES: List[str] = os.getenv("HYBRID_CHUNKER_STRATEGIES", "recursive,by_title").split(',')


# 임베딩 모델 설정 (예시)
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
# EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") # EnvApiKeyAdapter가 처리 가능


def create_app() -> FastAPI:
    """
    FastAPI 애플리케이션 인스턴스를 생성하고,
    시스템의 모든 구성 요소를 조립(Wiring)하여 의존성을 주입합니다.
    이 함수가 헥사고날 아키텍처의 컴포지션 루트 역할을 합니다.
    """
    print("--- Starting Application Assembly ---")

    # --- 1. 세컨더리 어댑터 인스턴스 생성 ---
    # 애플리케이션 코어(유스케이스)가 필요로 하는 외부 기능을 제공하는 어댑터들(출력 포트 구현체)을 생성하고 설정합니다.
    # 어댑터가 다른 어댑터나 외부 설정에 의존한다면 여기서 해당 의존성을 주입합니다.

    # API 키 관리 어댑터 (다른 어댑터들이 의존할 수 있음)
    apikey_adapter = EnvApiKeyAdapter()
    print("- Created EnvApiKeyAdapter instance.")

    # 파싱 어댑터 (Docling 구현체 - DocumentParsingPort)
    # __init__에 allowed_formats 등 설정 전달 (DoclingAdapter 상세 구현에서 확인)
    parser_adapter = DoclingParserAdapter(allowed_formats=DOCLING_ALLOWED_FORMATS)
    print("- Created DoclingParserAdapter instance.")

    # 청킹 어댑터 (Docling HybridChunker 구현체 또는 폴백 - TextChunkingPort)
    # __init__에 청킹 설정 전달 (DoclingChunkerAdapter 상세 구현에서 확인)
    chunker_adapter = DoclingChunkerAdapter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
    # HybridChunker 특정 설정 전달 예시:
    # chunker_adapter = DoclingChunkerAdapter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP, splitting_strategies=HYBRID_CHUNKER_STRATEGIES)
    print("- Created DoclingChunkerAdapter instance.")

    # 임베딩 어댑터 (BGE-M3 구현체 - EmbeddingGenerationPort)
    # __init__에 모델 설정 및 API 키 어댑터 주입 (BgeM3EmbedderAdapter 상세 구현에서 확인)
    # MilvusAdapter는 EmbeddingVector를 사용하므로, 임베딩 어댑터가 MilvusAdapter보다 먼저 생성되어야 합니다.
    embedder_adapter = BgeM3EmbedderAdapter(
        model_name=EMBEDDING_MODEL_NAME,
        device=EMBEDDING_DEVICE,
        api_key_port=apikey_adapter # 임베딩 어댑터가 API 키가 필요하다면 주입
    )
    print("- Created BgeM3EmbedderAdapter instance.")

    # 데이터 저장소 어댑터 (Milvus 구현체 - VectorDatabasePort)
    # __init__에 Milvus 접속 정보 주입 및 연결 시도 (MilvusAdapter 상세 구현에서 확인)
    persistence_adapter = None # 기본값 None
    try:
        persistence_adapter = MilvusAdapter(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            collection_name=MILVUS_COLLECTION,
            user=MILVUS_USER,
            password=MILVUS_PASSWORD,
            # MilvusAdapter __init__에 필요한 다른 파라미터 추가
        )
        print("- Created MilvusAdapter instance and attempted connection.")
    except Exception as e:
         print(f"--- WARNING: Failed to initialize MilvusAdapter. Persistence functionality may not work. --- Error: {e}")
         # DB 연결 실패 시 애플리케이션 시작을 중단할지, 아니면 경고만 남기고 계속 진행할지 정책 결정이 필요합니다.
         # 현재 코드는 경고만 남기고 persistence_adapter를 None으로 둡니다.
         # 유스케이스는 persistence_port가 필수이므로, 아래 유스케이스 생성 시 오류가 발생하거나,
         # 유스케이스가 None을 허용하도록 수정해야 합니다.
         # 현재 유스케이스 코드는 persistence_port가 필수이므로, persistence_adapter가 None이면 여기서 TypeError 발생합니다.


    # --- 2. 애플리케이션 계층 유스케이스 인스턴스 생성 ---
    # IngestDocumentUseCase는 DocumentProcessingInputPort를 구현하며, 필요한 출력 포트 구현체(세컨더리 어댑터)들을 주입받습니다.
    # 유스케이스의 __init__ 메서드에 정의된 파라미터에 맞춰 모든 어댑터를 주입합니다.

    # IngestDocumentUseCase는 persistence_port가 필수입니다. persistence_adapter가 None이면 유스케이스 생성 실패
    # MilvusAdapter 초기화가 성공했다면 아래 코드는 정상 작동합니다.
    ingest_use_case_instance = IngestDocumentUseCase(
        parser_port=parser_adapter,
        chunking_port=chunker_adapter,
        embedding_port=embedder_adapter,
        persistence_port=persistence_adapter, # <-- MilvusAdapter 인스턴스 주입 (성공했다면)
        api_key_port=apikey_adapter # 유스케이스 자체도 API Key Port가 필요하다면 주입
    )
    print("- Created IngestDocumentUseCase instance and injected dependencies.")


    # --- 3. FastAPI 애플리케이션 인스턴스 생성 ---
    app = FastAPI(
        title="RAG Hexagonal Document Ingestion System",
        description="Document ingestion and processing system based on Hexagonal Architecture, using Docling and Milvus.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    print("- Created FastAPI application instance.")

    # --- 4. 프라이머리 어댑터(FastAPI 라우터) 설정 및 애플리케이션에 포함 ---
    # 프라이머리 어댑터의 setup 함수에 유스케이스 인스턴스(입력 포트 구현체)를 주입하여 라우터 설정
    api_router = setup_api_routes(input_port=ingest_use_case_instance)
    print("- Configured API router with IngestDocumentUseCase.")

    # 설정된 라우터를 FastAPI 애플리케이션에 포함시켜 실제 엔드포인트가 활성화되도록 합니다.
    app.include_router(api_router)
    print("- Included API router in FastAPI app.")

    # --- 5. 기타 초기화 로직 (선택 사항) ---
    # @app.on_event("startup") 이벤트 핸들러 등록 등
    # 예: @app.on_event("startup")에서 Milvus 컬렉션 로드 등 초기 DB 작업 수행 가능
    # if persistence_adapter and hasattr(persistence_adapter, 'load_collection'): # 어댑터에 load_collection 메서드 필요
    #      @app.on_event("startup")
    #      async def load_milvus_collection():
    #          try:
    #              # 실제 pymilvus load_collection 호출 (MilvusAdapter 내부에 메서드 구현 필요)
    #              persistence_adapter.load_collection(collection_name=MILVUS_COLLECTION)
    #              print(f"Milvus collection '{MILVUS_COLLECTION}' loaded on startup.")
    #          except Exception as e:
    #              print(f"Error loading Milvus collection '{MILVUS_COLLECTION}' on startup: {e}")


    print("--- Application Assembly Complete ---")
    print("Application is ready.")

    # 조립이 완료된 FastAPI 애플리케이션 인스턴스 반환
    return app

# create_app 함수를 호출하여 애플리케이션 인스턴스 생성
# MilvusAdapter 초기화 실패 시 create_app 내에서 예외가 발생하고 함수가 중단될 수 있습니다.
try:
    app = create_app()
except Exception as e: # create_app 실행 중 발생하는 예외 처리
    print(f"--- FATAL ERROR: Application startup failed during assembly --- Error: {e}")
    app = None # 앱 인스턴스 생성 실패


# --- 애플리케이션 실행 ---
# create_app이 성공하고 app 인스턴스가 생성된 경우에만 서버 실행
if __name__ == "__main__" and app is not None:
    print("\n--- Starting FastAPI application server ---")
    # uvicorn.run() 함수 호출. 호스트와 포트 설정. reload=True는 개발용.
    try:
        # uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) # app 인스턴스 직접 전달
        # 또는 모듈 경로 문자열 전달 방식 (main:app)
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) # 개발 모드 예시 (파일 이름:모듈 이름 형태)
    except Exception as e:
         print(f"Error running uvicorn server: {e}")
    print("--- FastAPI application server stopped ---")
elif app is None:
    print("\nFastAPI application server did not start due to previous assembly errors.")