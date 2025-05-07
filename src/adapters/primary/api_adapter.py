# src/adapters/primary/api_adapter.py

from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from typing import List

# 도메인 모델 임포트 (domain/models.py 에 정의될 모델)
# RawDocument는 외부 입력을 애플리케이션 모델로 변환할 때 사용됩니다.
# DocumentChunk는 유스케이스로부터 반환받아 외부에 제공할 때 사용됩니다.
from domain.models import RawDocument, DocumentChunk

# 입력 포트 임포트 (애플리케이션 계층의 유스케이스가 이 포트를 구현합니다)
# 프라이머리 어댑터는 이 포트를 통해 애플리케이션 코어와 통신합니다.
from ports.input_ports import DocumentProcessingInputPort

# APIRouter 인스턴스 생성
# 이 라우터가 특정 URL 경로에 대한 요청을 처리합니다.
router = APIRouter()

# --- 중요: 의존성 주입 설정 ---
# 이 부분은 FastAPI의 Depends 시스템 또는 다른 DI 프레임워크에 맞게
# main.py 또는 앱 초기화 부분에서 설정될 것입니다.
# 여기서는 setup_api_routes 함수를 사용하여 input_port 인스턴스를 주입받는
# 방식을 예시로 보여줍니다. 실제 프로덕션 코드에서는 FastAPI의 Depends를
# 사용하는 것이 일반적입니다.

# 예시: DocumentProcessingInputPort 구현체(IngestDocumentUseCase 인스턴스)를
# FastAPI의 Depends 기능을 통해 주입받는다고 가정하는 함수 (실제 구현 필요)
# def get_document_processing_input_port_dependency() -> DocumentProcessingInputPort:
#     """
#     FastAPI Depends를 위한 함수. 실제 IngestDocumentUseCase 인스턴스를 반환하도록
#     main.py의 DI 컨테이너에서 설정됩니다.
#     """
#     # 이 함수는 main.py에서 오버라이드되거나 설정됩니다.
#     raise NotImplementedError("Dependency not configured in Composition Root (main.py)")

# 아래 setup_api_routes 함수는 DI 설정을 외부로 위임하는 한 가지 방법입니다.
def setup_api_routes(input_port: DocumentProcessingInputPort) -> APIRouter:
    """
    FastAPI 라우터를 설정하고, 이 라우터가 의존할 DocumentProcessingInputPort 구현체를 주입받습니다.

    Args:
        input_port: DocumentProcessingInputPort 인터페이스를 구현한 인스턴스
                    (대부분 IngestDocumentUseCase 인스턴스가 될 것입니다).

    Returns:
        설정이 완료된 FastAPI APIRouter 인스턴스.
    """

    @router.post(
        "/ingest-document", # 문서 처리를 시작하는 API 엔드포인트 경로
        response_model=List[DocumentChunk], # API 호출 성공 시 반환될 데이터 모델 (자동 직렬화/역직렬화)
        status_code=status.HTTP_201_CREATED # 성공적인 리소스 생성(문서 처리 결과)을 나타내는 HTTP 상태 코드
    )
    async def ingest_document_endpoint(
        file: UploadFile = File(...) # FastAPI를 통해 업로드된 파일 데이터를 자동으로 바인딩
        # 만약 FastAPI Depends 방식을 사용한다면 여기에 주입 코드가 들어갑니다:
        # input_port: DocumentProcessingInputPort = Depends(get_document_processing_input_port_dependency)
    ):
        """
        외부 클라이언트로부터 문서를 업로드 받아 애플리케이션 코어의 문서 처리 프로세스를 시작시키는 API 엔드포인트입니다.
        이 어댑터는 실제 파싱이나 청킹 로직을 수행하지 않고, 해당 작업을 애플리케이션 코어에 위임합니다.
        """
        if not file.filename:
             raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file uploaded"
             )

        try:
            # 1. 외부에서 들어온 데이터 형식(FastAPI UploadFile)을 애플리케이션 코어가 이해하는
            #    도메인 모델 형식(RawDocument)으로 변환합니다. 이것이 어댑터의 핵심 역할 중 하나입니다.
            content = await file.read()
            # 실제로는 파일 크기, 형식 등 추가적인 사전 검증 로직이 여기에 포함될 수 있습니다.

            raw_document = RawDocument(
                content=content,
                metadata={
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "size": len(content) # 예시 메타데이터: 파일 크기 추가
                    # 추후 요청 시간, 요청 사용자 ID 등 추적/로깅에 유용한 메타데이터를 추가할 수 있습니다.
                }
            )

            # 2. 변환된 도메인 모델을 애플리케이션 코어의 입력 포트를 통해 전달하여
            #    문서 처리 프로세스(파싱 -> 청킹 -> ...)를 실행하도록 요청합니다.
            #    setup_api_routes 함수를 사용한 경우, input_port는 함수의 인자로 이미 전달받은 상태입니다.
            processed_chunks = input_port.execute(raw_document)

            # 3. 애플리케이션 코어로부터 반환된 결과(List[DocumentChunk])를
            #    외부 클라이언트에게 응답할 형식으로 변환합니다.
            #    FastAPI의 response_model=List[DocumentChunk] 설정이 이 변환(파이단틱 모델 직렬화)을 자동으로 처리해 줍니다.
            return processed_chunks

        except Exception as e:
            # 처리 중 발생한 예외를 잡아서 적절한 HTTP 오류 응답으로 변환합니다.
            # 실제 시스템에서는 예외의 종류에 따라 더 세분화된 HTTP 상태 코드를 반환해야 합니다.
            # (예: 데이터 유효성 오류 -> 422, 권한 오류 -> 403, 찾을 수 없음 -> 404 등)
            print(f"Error during document ingestion: {e}") # 서버 로그에 오류 상세 기록
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # 일반적인 서버 내부 오류 상태 코드
                detail=f"Failed to process document due to an internal error: {e}" # 오류 상세 메시지 (디버깅용 또는 사용자용)
            )

    # 설정이 완료된 APIRouter 인스턴스를 반환하여 main.py에서 FastAPI 앱에 포함시킬 수 있도록 합니다.
    return router

# --- main.py 에서 사용하는 예시 (주석 처리) ---
# from fastapi import FastAPI
# from adapters.primary.api_adapter import setup_api_routes # setup 함수 임포트
#
# # --- 1. 세컨더리 어댑터 인스턴스 생성 (실제 Docling, BGE-M3 등을 감싸는 객체) ---
# # from adapters.secondary.docling_parser_adapter import DoclingParserAdapter
# # from adapters.secondary.docling_chunker_adapter import DoclingChunkerAdapter
# # from adapters.secondary.bge_m3_embedder_adapter import BgeM3EmbedderAdapter
# # from adapters.secondary.env_apikey_adapter import EnvApiKeyAdapter
# # parser_adapter = DoclingParserAdapter(...)
# # chunker_adapter = DoclingChunkerAdapter(...)
# # embedder_adapter = BgeM3EmbedderAdapter(...)
# # apikey_adapter = EnvApiKeyAdapter(...)
#
# # --- 2. 애플리케이션 계층 유스케이스 인스턴스 생성 (입력 포트 구현체) ---
# # 유스케이스는 필요한 출력 포트 구현체(세컨더리 어댑터 인스턴스)를 주입받습니다.
# # from application.use_cases import IngestDocumentUseCase
# # ingest_use_case_instance = IngestDocumentUseCase(
# #     parser_port=parser_adapter,
# #     chunking_port=chunker_adapter,
# #     embedding_port=embedder_adapter, # 유스케이스가 임베딩도 포함한다면
# #     api_key_port=apikey_adapter # 필요한 경우
# # )
#
# # --- 3. FastAPI 애플리케이션 인스턴스 생성 ---
# # app = FastAPI()
#
# # --- 4. 프라이머리 어댑터 라우터 설정 및 애플리케이션에 포함 ---
# # setup 함수에 유스케이스 인스턴스를 전달하여 라우터 설정
# # api_router = setup_api_routes(input_port=ingest_use_case_instance)
# # 설정된 라우터를 FastAPI 앱에 포함
# # app.include_router(api_router)
#
# # --- 5. 앱 실행 (uvicorn 등 사용) ---
# # 예: uvicorn main:app --reload