# src/domain/models.py

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# RawDocument: 시스템에 처음 입력되는 원본 문서 데이터
@dataclass
class RawDocument:
    """
    처리되지 않은 원본 문서 데이터를 나타내는 도메인 모델.
    """
    content: bytes # 문서의 바이너리/텍스트 내용
    metadata: Dict[str, Any] = field(default_factory=dict) # 파일명, 타입 등 관련 메타데이터

# ParsedDocument: 파싱 과정을 거쳐 구조화된 문서 데이터
@dataclass
class ParsedDocument:
    """
    파싱 과정을 거쳐 텍스트 추출 및 기본 구조 정보가 포함된 문서 모델.
    """
    content: str # 추출된 텍스트 내용
    metadata: Dict[str, Any] = field(default_factory=dict) # 원본 메타데이터 + 파싱 중 얻은 메타데이터 (예: 제목, 저자)

# DocumentChunk: 검색 및 임베딩을 위한 문서의 작은 단위
@dataclass
class DocumentChunk:
    """
    문서의 작은 의미 단위(청크)를 나타내는 도메인 모델.
    """
    content: str # 청크의 텍스트 내용
    metadata: Dict[str, Any] = field(default_factory=dict) # 원본/파싱된 메타데이터 + 청크별 메타데이터 (예: 페이지 번호, 청크 ID)

# EmbeddingVector: 문서 청크의 벡터 표현
@dataclass
class EmbeddingVector:
    """
    문서 청크의 임베딩 벡터 표현과 관련 메타데이터를 나타내는 도메인 모델.
    """
    vector: List[float] # 임베딩 벡터 값 리스트 (dense vector)
    metadata: Dict[str, Any] = field(default_factory=dict) # 해당 임베딩이 파생된 청크 정보 등
    sparse_vector: Optional[Dict[int, float]] = None # BGE-M3의 스파스 벡터 (indices와 values 정보)