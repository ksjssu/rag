# src/ports/input_ports.py

from abc import ABC, abstractmethod
from typing import List

# 도메인 모델 임포트 (domain/models.py 에 정의될 모델)
from domain.models import RawDocument, DocumentChunk # Assuming these models exist

class DocumentProcessingInputPort(ABC):
    """
    문서 처리(가져오기) 프로세스를 시작하기 위한 애플리케이션 코어의 입력 포트 정의.
    이 포트는 애플리케이션 계층의 유스케이스에 의해 구현됩니다.
    """

    @abstractmethod
    def execute(self, raw_document: RawDocument) -> List[DocumentChunk]:
        """
        주어진 RawDocument를 파싱, 청킹하는 문서 가져오기 프로세스를 실행합니다.

        Args:
            raw_document: 처리할 원본 문서 데이터 (RawDocument 도메인 모델).

        Returns:
            처리된 DocumentChunk 객체들의 리스트를 반환합니다.
            (주: 이 포트의 반환 타입은 유스케이스가 최종적으로 외부에 제공하는 결과에 따라 달라질 수 있습니다.
             여기서는 파싱 및 청킹 결과인 청크 목록을 반환하는 것으로 가정합니다.)
        """
        pass

# 필요하다면 다른 입력 포트들도 여기에 정의할 수 있습니다 (예: GetDocumentStatusInputPort 등)
# class OtherInputPort(ABC):
#     @abstractmethod
#     def perform_action(self, data: Any) -> Any:
#         pass