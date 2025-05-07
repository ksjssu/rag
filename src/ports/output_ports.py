# src/ports/output_ports.py

from abc import ABC, abstractmethod
from typing import List

# 도메인 모델 임포트 (domain/models.py 에 정의될 모델)
# 이 포트들이 입/출력 데이터 타입으로 사용할 도메인 모델들입니다.
from domain.models import RawDocument, ParsedDocument, DocumentChunk, EmbeddingVector

class DocumentParsingPort(ABC):
    """
    애플리케이션 코어(유스케이스)가 RawDocument를 ParsedDocument로 파싱하기 위해 사용하는 출력 포트.
    이 포트는 외부 파싱 어댑터(예: Docling 파서 어댑터)에 의해 구현됩니다.
    """

    @abstractmethod
    def parse(self, raw_document: RawDocument) -> ParsedDocument:
        """
        주어진 원본 문서를 파싱하여 구조화된 문서 모델을 반환합니다.

        Args:
            raw_document: 파싱할 원본 문서 데이터 (RawDocument 도메인 모델).

        Returns:
            파싱된 문서 데이터 모델 (ParsedDocument 도메인 모델).
        """
        pass

class TextChunkingPort(ABC):
    """
    애플리케이션 코어(유스케이스)가 ParsedDocument를 DocumentChunk 목록으로 청킹하기 위해 사용하는 출력 포트.
    이 포트는 외부 청킹 어댑터(예: Docling 청커 어댑터)에 의해 구현됩니다.
    """

    @abstractmethod
    def chunk(self, parsed_document: ParsedDocument) -> List[DocumentChunk]:
        """
        주어진 파싱된 문서를 의미있는 청크들로 나눕니다.

        Args:
            parsed_document: 청킹할 파싱된 문서 데이터 (ParsedDocument 도메인 모델).

        Returns:
            DocumentChunk 객체들의 리스트.
        """
        pass

class EmbeddingGenerationPort(ABC):
    """
    애플리케이션 코어(유스케이스)가 DocumentChunk 목록을 임베딩 벡터로 변환하기 위해 사용하는 출력 포트.
    이 포트는 외부 임베딩 어댑터(예: BGE-M3 임베더 어댑터)에 의해 구현됩니다.
    """
    @abstractmethod
    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[EmbeddingVector]:
        """
        주어진 문서 청크들에 대한 임베딩 벡터를 생성합니다.

        Args:
            chunks: 임베딩을 생성할 DocumentChunk 객체들의 리스트.

        Returns:
            생성된 EmbeddingVector 객체들의 리스트 (일반적으로 chunks 리스트와 동일한 순서 및 개수).
        """
        pass

class ApiKeyManagementPort(ABC):
    """
    애플리케이션 코어(유스케이스 또는 다른 서비스)가 외부 서비스 사용에 필요한 API 키 등을
    가져오기 위해 사용하는 출력 포트. 이 포트는 환경 변수 어댑터 등에 의해 구현됩니다.
    """
    @abstractmethod
    def get_api_key(self, service_name: str) -> str:
        """
        주어진 서비스 이름에 해당하는 API 키를 조회합니다.

        Args:
            service_name: API 키가 필요한 외부 서비스의 이름 (예: "OPENAI", "DO CLING", "BGE_M3").

        Returns:
            해당 서비스의 API 키 문자열.
            키가 없을 경우 예외를 발생시키거나 적절한 오류 처리를 구현체에서 수행합니다.
        """
        pass
    
    
class VectorDatabasePort(ABC):
    """
    애플리케이션 코어(유스케이스)가 문서 청크와 임베딩 벡터를
    벡터 데이터베이스에 저장하기 위해 사용하는 출력 포트.

    이 포트는 실제 벡터 데이터베이스 어댑터(예: MilvusAdapter, QdrantAdapter 등)에 의해 구현됩니다.
    """

    @abstractmethod
    def save_document_data(self, chunks: List[DocumentChunk], embeddings: List[EmbeddingVector]) -> None:
        """
        문서 청크 목록과 해당하는 임베딩 벡터 목록을 벡터 데이터베이스에 저장합니다.

        청크와 임베딩은 1:1로 매핑되어 있으며, 같은 순서로 전달된다고 가정합니다.
        임베딩과 함께 저장소에 필요한 메타데이터는 DocumentChunk와 EmbeddingVector
        모델의 metadata 필드에 포함되어야 합니다.

        Args:
            chunks: 저장할 DocumentChunk 객체들의 리스트.
            embeddings: 저장할 EmbeddingVector 객체들의 리스트.

        Raises:
            VectorDatabaseError: 데이터 저장 중 벡터 데이터베이스 관련 오류 발생 시
                               (이 포트를 구현하는 어댑터에서 이 예외를 정의하고 발생시킬 수 있습니다).
        """
        pass

    # --- 검색 기능도 VectorDatabasePort의 역할이 될 수 있습니다 (필요시 추가) ---
    # RAG 시스템의 검색 단계에서 벡터 DB 조회가 필요하므로, 검색 포트와 동일하거나 여기에 검색 메서드를 추가합니다.
    # 현재는 Ingestion 워크플로우에 집중하므로 저장 메서드만 정의합니다.
    # @abstractmethod
    # def search_similar_vectors(self, query_embedding: EmbeddingVector, top_k: int) -> List[DocumentChunk]:
    #     """
    #     주어진 쿼리 임베딩 벡터와 유사한 벡터를 데이터베이스에서 검색하고 해당 청크를 반환합니다.
    #
    #     Args:
    #         query_embedding: 검색에 사용할 임베딩 벡터.
    #         top_k: 가장 유사한 상위 k개의 결과를 가져옵니다.
    #
    #     Returns:
    #         검색된 관련 DocumentChunk 객체들의 리스트.
    #         (검색 결과에서 DocumentChunk를 재구성하거나, 저장 시 청크 내용을 DB에 함께 저장해야 함)
    #     Raises:
    #         VectorDatabaseError: 검색 중 오류 발생 시.
    #     """
    #     pass