# src/application/use_cases.py

import logging
from typing import List, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# --- 도메인 모델 임포트 ---
# 기존 오류 발생 라인: from .domain.models import RawDocument, ParsedDocument, DocumentChunk, EmbeddingVector
# 수정: 부모 패키지(src)로 한 단계 올라간 후(..) domain 서브 패키지로 내려가 models 임포트
from ..domain.models import RawDocument, ParsedDocument, DocumentChunk, EmbeddingVector # <-- 상대 경로 임포트 수정 (..domain...)

# --- 포트 임포트 ---
# use_cases.py는 application 서브 패키지에 있고, ports 서브 패키지는 src 바로 아래 있으므로 ports도 형제 패키지입니다.
# 따라서 ports 임포트도 ..ports 형태로 수정해야 합니다.
# 기존: from .ports.input_ports import DocumentProcessingInputPort
# 기존: from .ports.output_ports import ( ... )

from ..ports.input_ports import DocumentProcessingInputPort # <-- 상대 경로 임포트 수정 (..ports...)
from ..ports.output_ports import ( # <-- 상대 경로 임포트 수정 (..ports...)
    DocumentParsingPort,
    TextChunkingPort,
    EmbeddingGenerationPort,
    ApiKeyManagementPort,
    VectorDatabasePort,
    # 필요한 다른 저장소 포트가 있다면 여기에 추가
)

# 유스케이스는 입력 포트 인터페이스를 구현합니다.
class IngestDocumentUseCase(DocumentProcessingInputPort):
    # ... (나머지 use_cases.py 코드는 그대로 유지) ...

    def __init__(
        self,
        parser_port: DocumentParsingPort,       # 파싱 기능 제공 포트
        chunking_port: TextChunkingPort,       # 청킹 기능 제공 포트
        embedding_port: EmbeddingGenerationPort, # 임베딩 기능 제공 포트
        persistence_port: VectorDatabasePort, # <-- VectorDatabasePort 의존성 추가
        api_key_port: Optional[ApiKeyManagementPort] = None # 필요하다면 API 키 관리 포트 (None 허용)
    ):
        """
        IngestDocumentUseCase 초기화 및 필요한 출력 포트 구현체들을 주입받습니다.

        Args:
            parser_port: DocumentParsingPort 인터페이스를 구현한 어댑터 인스턴스.
            chunking_port: TextChunkingPort 인터페이스를 구현한 어댑터 인스턴스.
            embedding_port: EmbeddingGenerationPort 인터페이스를 구현한 어댑터 인스턴스.
            persistence_port: VectorDatabasePort 인터페이스를 구현한 어댑터 인스턴스 (예: MilvusAdapter).
            api_key_port: ApiKeyManagementPort 인터페이스를 구현한 어댑터 인스턴스 (선택 사항).
        """
        self._parser_port: DocumentParsingPort = parser_port
        self._chunking_port: TextChunkingPort = chunking_port
        self._embedding_port: EmbeddingGenerationPort = embedding_port
        self._persistence_port: VectorDatabasePort = persistence_port # <-- 주입받은 저장소 포트 저장
        self._api_key_port: Optional[ApiKeyManagementPort] = api_key_port
        logger.info("IngestDocumentUseCase initialized with parser, chunking, embedding, and persistence ports.")

    # DocumentProcessingInputPort 인터페이스의 execute 메서드를 구현합니다.
    # InputPort 정의는 List[DocumentChunk]를 반환하므로, 현재는 그대로 유지합니다.
    # 임베딩 결과는 내부적으로 사용(예: 저장소 포트 호출)하고 청크 목록을 반환합니다.
    def execute(self, raw_document: RawDocument) -> List[DocumentChunk]:
        """
        문서 가져오기 프로세스를 실행하는 핵심 메서드입니다.

        RawDocument를 입력받아 순차적으로 파싱, 청킹, 임베딩 과정을 진행하고
        생성된 청크와 임베딩을 저장소 포트를 통해 저장합니다.

        Args:
            raw_document: 처리할 원본 문서 데이터.

        Returns:
            처리 결과인 DocumentChunk 객체들의 리스트.
            (참고: 임베딩과 청크 데이터는 이 메서드 내부에서 저장소 포트를 통해 저장됩니다.
             반환 값은 저장된 청크 목록으로 유지합니다.)
        """
        # 초기 로그
        logger.info(f"[USECASE] 시작: {raw_document.metadata.get('filename', 'unknown')}")
        
        try:
            # None 검사 추가
            if raw_document is None:
                logger.error("IngestDocumentUseCase: raw_document is None")
                return []
            
            if not hasattr(raw_document, 'content') or raw_document.content is None:
                logger.error("IngestDocumentUseCase: raw_document.content is None or missing")
                return []
            
            # 1. 파싱 (파싱 포트 호출)
            parsed_document = self._parser_port.parse(raw_document)
            
            # None 검사 추가
            if parsed_document is None:
                logger.error("IngestDocumentUseCase: parsed_document is None")
                return []
            
            if not hasattr(parsed_document, 'content') or parsed_document.content is None:
                logger.error("IngestDocumentUseCase: parsed_document.content is None or missing")
                return []
            
            logger.info(f"[USECASE] 파싱 완료: 길이 {len(parsed_document.content)}")

            # 2. 청킹 (청킹 포트 호출)
            chunks = self._chunking_port.chunk(parsed_document)
            logger.info(f"[USECASE] 청킹 완료: 개수 {len(chunks)}")
            
            # 테이블 및 처리 유형 관련 청크 확인
            table_chunks = 0
            image_chunks = 0
            text_chunks = 0
            
            for chunk in chunks:
                chunk_type = "text"  # 기본값
                if 'source' in chunk.metadata:
                    chunk_type = chunk.metadata['source']
                
                if chunk_type == "table":
                    table_chunks += 1
                    # 테이블 내용이 content에 제대로 포함되는지 확인
                    if table_chunks <= 2:  # 처음 2개 테이블 청크만 로그에 출력
                        logger.info(f"[TABLE CHUNK #{table_chunks}] 내용 샘플: {chunk.content[:100]}...")
                elif chunk_type == "image":
                    image_chunks += 1
                else:
                    text_chunks += 1
            
            logger.info(f"[USECASE] 청크 유형 통계: 텍스트={text_chunks}, 테이블={table_chunks}, 이미지={image_chunks}")

            # 청크가 없다면 임베딩 및 저장 단계 스킵
            if not chunks:
                 logger.info("IngestDocumentUseCase: No chunks generated, skipping embedding and persistence.")
                 return [] # 청크가 없으므로 빈 리스트 반환

            # 3. 임베딩 (임베딩 포트 호출)
            embeddings: List[EmbeddingVector] = self._embedding_port.generate_embeddings(chunks)
            logger.info(f"IngestDocumentUseCase: Generated {len(embeddings)} embeddings.")

            # 임베딩 결과가 청크와 1:1 매핑되는지 확인 (어댑터가 이를 보장한다고 가정)
            if len(chunks) != len(embeddings):
                 logger.warning(f"Warning: Number of chunks ({len(chunks)}) and embeddings ({len(embeddings)}) do not match. Data inconsistency.")
                 # 여기서 어떻게 처리할지 결정 (저장 중단, 경고 후 저장 시도 등)
                 # 일단 경고만 남기고 저장을 시도하되, 저장 어댑터에서 오류가 나도록 합니다.
                 # raise ApplicationError("Chunk and embedding count mismatch") # 또는 특정 예외 발생

            # --- 4. 데이터 저장 (데이터 저장소 포트 호출) ---
            # 생성된 청크와 임베딩 목록을 VectorDatabasePort 구현체에게 전달하여 저장 요청
            logger.info(f"IngestDocumentUseCase: Saving {len(chunks)} chunks and {len(embeddings)} embeddings via persistence port...")
            self._persistence_port.save_document_data(chunks, embeddings) # <-- ★ 저장 메서드 호출 ★
            logger.info("IngestDocumentUseCase: Chunks and embeddings successfully sent for persistence.")


            # 현재 DocumentProcessingInputPort의 정의에 따라 처리된 청크 목록을 반환합니다.
            # 저장까지 포함하는 Ingestion 프로세스의 최종 결과로서 청크 목록을 반환하는 것이 일반적입니다.
            return chunks

        except Exception as e:
             # 파싱, 청킹, 임베딩, 저장 등 유스케이스 실행 중 발생한 예외 처리
             # 세컨더리 어댑터에서 VectorDatabaseError 등 어댑터 특정 예외를 발생시키면 여기서 잡아서 처리합니다.
             # 유스케이스는 이러한 하위 레벨 예외를 도메인 또는 애플리케이션 레벨의 예외로 래핑하여
             # 상위 호출자(프라이머리 어댑터)에게 다시 발생시키는 것이 좋습니다.
             logger.error(f"IngestDocumentUseCase: An error occurred during processing - {e}")
             # 예외를 상위 호출자(프라이머라 어댑터)로 전파합니다.
             raise

# 다른 유스케이스들...