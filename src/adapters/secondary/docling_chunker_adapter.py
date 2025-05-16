# src/adapters/secondary/docling_chunker_adapter.py

import logging
from typing import List, Dict, Any, Iterable, Optional, Union
import re

# Configure logging
logger = logging.getLogger(__name__)

# config 불러오기 - 설정 참조를 위해 추가
from src.config import settings

# --- Docling 라이브러리 임포트 ---
# src/adapters/secondary/docling_chunker_adapter.py

# --- Docling 라이브러리 임포트 ---
# 제공된 디렉토리 목록에 기반하여 정확한 임포트 경로로 수정합니다.
# 최상위 패키지는 'docling' 입니다.
try:
    # Docling 청킹을 위한 핵심 클래스 임포트
    # from docling_core.transforms.chunker import HybridChunker # 이전 시도 (실패)
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker # <-- 정확한 경로: docling/transforms/chunker/hybrid_chunker.py

    # HybridChunker가 의존하는 내부 클래스 임포트 (초기화 파라미터 타입 힌트용)
    # from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer # 이전 시도 (실패)
    from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer # <-- 정확한 경로: docling/transforms/chunker/tokenizer/base.py

    # Docling 내부 문서 객체 타입 정의 (DoclingDocument)
    # from docling_core.types import DoclingDocument # 이전 시도 (실패)
    from docling_core.types.doc import DoclingDocument # <-- 정확한 경로: docling/types/doc/document.py (listing 참고)


    _docling_chunker_available = True
    logger.info("Docling chunking library (HybridChunker) imported successfully.")
except ImportError as e: # 임포트 실패 시 발생하는 예외 메시지를 출력하도록 수정
    logger.warning(f"Warning: Docling library import failed. Import error: {e}") # <-- 실제 임포트 오류 메시지 출력
    logger.warning("DoclingChunkerAdapter will use simple line splitting.")
    _docling_chunker_available = False
    # --- Docling 클래스가 없을 경우 에러 방지를 위한 더미 클래스 정의 ---
    # (기존 더미 클래스 정의는 그대로 유지)
    logger.info("   Using dummy Docling chunker classes.")
    # Dummy 클래스 정의들...

    # (이전에 정의했던 Dummy HybridChunker, MockDocChunkResult 클래스 등)

    # ★★★ 여기부터 추가 ★★★
    # 임포트 실패 시 BaseTokenizer 이름이 정의되도록 더미 클래스 추가
    # HybridChunker 코드를 보면 .get_max_tokens()와 .count_tokens() 메서드가 사용됩니다.
    # 더미 클래스에 이 메서드들을 추가하여 NameError를 방지합니다.
    class BaseTokenizer:
        def __init__(
            self,
            # ... tokenizer, max_tokens, merge_peers ...
            fallback_chunk_size: int = 1000,  # <--- 이름이 fallback_chunk_size
            fallback_chunk_overlap: int = 200,  # <--- 이름이 fallback_chunk_overlap
            # ... 기타 파라미터 ...
        ):
            pass

        # HybridChunker 코드에서 사용되는 메서드들을 더미로 정의
        def get_max_tokens(self) -> int:
            logger.info("(Dummy BaseTokenizer.get_max_tokens)")
            # 기본값 또는 placeholder 값 반환
            return 1000  # 예시: 기본 최대 토큰 수 1000 반환

        def count_tokens(self, text: str) -> int:
            logger.info("(Dummy BaseTokenizer.count_tokens)")
            # 간단하게 문자열 길이 반환 (토큰 수를 흉내)
            return len(text)

    # ★★★ 여기까지 추가 ★★★

    # (나머지 더미 클래스들: HybridChunker, MockDocChunkResult 등은 이전 코드에서 그대로 유지되어야 합니다.)


# --- 어댑터 특정 예외 정의 ---
# 청킹 과정에서 발생하는 오류를 나타내기 위한 어댑터 레벨의 예외
class ChunkingError(Exception):
    """Represents an error during the text chunking process."""
    pass

# ... (나머지 DoclingChunkerAdapter 클래스 코드 계속) ...


from ports.output_ports import TextChunkingPort # 구현할 포트 임포트
from domain.models import ParsedDocument, DocumentChunk # 입/출력 도메인 모델 임포트
from typing import List, Dict, Any, Iterable, Optional, Union # Iterable, Optional, Union 임포트


# src/adapters/secondary/docling_chunker_adapter.py

# ... (imports and dummy classes) ...

class DoclingChunkerAdapter(TextChunkingPort):
    """
    Docling 라이브러리(HybridChunker)를 사용하여 TextChunkingPort를 구현하는 어댑터.
    ParsedDocument의 메타데이터에서 Docling 내부 문서 객체를 추출하여 청킹합니다.
    """
    def __init__(
        self,
        # HybridChunker 초기화에 필요한 파라미터들을 __init__에서 받습니다.
        tokenizer: Union[str, BaseTokenizer] = "sentence-transformers/all-MiniLM-L6-v2",
        max_tokens: Optional[int] = None,
        merge_peers: bool = True,
        # 기타 HybridChunker 초기화 파라미터가 있다면 여기에 추가

        # 기본값을 config.py의 설정값과 일치시킴
        chunk_size: int = None,  # 기본값을 None으로 설정하여 settings에서 가져오도록 함
        chunk_overlap: int = None,  # 기본값을 None으로 설정하여 settings에서 가져오도록 함
    ):
        """
        DoclingChunkerAdapter 초기화. Docling HybridChunker 인스턴스를 생성하거나 폴백 설정을 가집니다.

        Args:
            tokenizer: Docling HybridChunker에 사용될 토크나이저 설정.
            max_tokens: Docling HybridChunker의 최대 토큰 수 설정.
            merge_peers: Docling HybridChunker의 merge_peers 설정.
            # 기타 Docling HybridChunker 초기화 파라미터들 (Docling 문서 확인 필요)
            chunk_size: Docling Chunker 또는 폴백 청킹 크기. 기본값은 settings.CHUNK_SIZE
            chunk_overlap: Docling Chunker 또는 폴백 청킹 오버랩. 기본값은 settings.CHUNK_OVERLAP
        """
        # config에서 기본값 가져오기
        if chunk_size is None:
            chunk_size = settings.CHUNK_SIZE
        if chunk_overlap is None:
            chunk_overlap = settings.CHUNK_OVERLAP

        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size.")

        self._chunker: Optional[HybridChunker] = None
        # --- 인스턴스 변수 이름도 통일 ---
        self._chunk_size = chunk_size  # <-- 인스턴스 변수 이름도 _chunk_size로 변경
        self._chunk_overlap = chunk_overlap  # <-- 인스턴스 변수 이름도 _chunk_overlap로 변경

        # settings 객체 유지
        self._settings = settings

        # Docling HybridChunker 초기화에 필요한 파라미터들을 저장
        self._chunker_init_params: Dict[str, Any] = {
            'tokenizer': tokenizer,
            'max_tokens': max_tokens,
            'merge_peers': merge_peers,
            # 만약 HybridChunker 생성자가 chunk_size, overlap을 받는다면 여기에 추가
            # 'chunk_size': chunk_size, # <-- __init__ 파라미터 값을 전달
            # 'overlap': chunk_overlap, # <-- __init__ 파라미터 값을 전달
            # ... other params ...
        }

        if _docling_chunker_available:
            logger.info("DoclingChunkerAdapter: Initializing Docling HybridChunker...")
            try:
                # --- Docling HybridChunker 인스턴스 생성 ★ 실제 초기화 ★ ---
                # 제공된 HybridChunker 코드의 __init__ (Pydantic BaseModel)에 정의된 파라미터들을 전달
                self._chunker = HybridChunker(
                    **self._chunker_init_params  # 저장된 파라미터 딕셔너리 언팩하여 전달
                )
                logger.info("DoclingChunkerAdapter: Docling HybridChunker initialized successfully.")
            except Exception as e:  # Docling Chunker 초기화 중 발생할 수 있는 예외 처리
                logger.error(f"Error initializing Docling HybridChunker: {e}")
                self._chunker = None  # 초기화 실패 시 None
                # 초기화 실패 시 ChunkingError 예외를 발생시켜 앱 시작 중단 고려
                # raise ChunkingError(f"Failed to initialize Docling HybridChunker: {e}") from e

        if self._chunker is None:
            logger.warning(f"DoclingChunkerAdapter: Docling HybridChunker not available or failed to initialize. Will use simple quantitative chunking (size={self._chunk_size}, overlap={self._chunk_overlap}).")

    def chunk(self, parsed_document: ParsedDocument) -> List[DocumentChunk]:
        """
        ParsedDocument를 Docling HybridChunker로 청킹합니다.
        """
        logger.info("DoclingChunkerAdapter: Starting chunking process...")

        self._debug_parsed_content(parsed_document)

        chunks: List[DocumentChunk] = []
        # 파서 어댑터에서 ParsedDocument 메타데이터에 담아 전달한 Docling 내부 문서 객체를 추출
        # '__internal_docling_document__' 키는 파서 어댑터 코드와 일치해야 합니다.
        docling_internal_document = parsed_document.metadata.get('__internal_docling_document__')

        # --- Docling 청킹 로직 또는 폴백 로직 실행 ---

        if self._chunker and docling_internal_document:  # Docling 청커 인스턴스가 유효하고 내부 객체도 있다면 실제 청킹 실행
            logger.info("DoclingChunkerAdapter: Using configured Docling HybridChunker.")
            try:
                # --- ★★★ 실제 Docling 라이브러리 청킹 기능을 호출하는 부분 ★★★ ---
                logger.info("Calling self._chunker.chunk()...")
                docling_chunk_results_iterator: Iterable = self._chunker.chunk(  # <--- ▶︎▶︎▶︎ 실제 호출 라인! ◀︎◀︎◀︎
                    docling_internal_document,
                )
                logger.info("Received Docling chunking results iterator.")

                # --- Docling 청킹 결과 목록을 순회하며 처리 ---
                logger.info("Processing Docling chunking results iterator...")
                base_metadata = parsed_document.metadata.copy()
                base_metadata.pop('__internal_docling_document__', None)

                chunk_index = 0
                if isinstance(docling_chunk_results_iterator, Iterable):
                    for docling_chunk_result in docling_chunk_results_iterator:
                        chunk_content = ""
                        if hasattr(docling_chunk_result, 'text') and isinstance(docling_chunk_result.text, str):
                            chunk_content = docling_chunk_result.text
                        elif hasattr(docling_chunk_result, 'get_text'):
                            chunk_content = docling_chunk_result.get_text() or ""

                        current_chunk_metadata = base_metadata.copy()
                        current_chunk_metadata["chunk_index"] = chunk_index

                        docling_meta_object = None
                        if hasattr(docling_chunk_result, 'meta'):
                            docling_meta_object = docling_chunk_result.meta

                        if docling_meta_object:
                            if hasattr(docling_meta_object, 'headings'):
                                current_chunk_metadata['headings'] = docling_meta_object.headings
                            if hasattr(docling_meta_object, 'captions'):
                                current_chunk_metadata['captions'] = docling_meta_object.captions
                            if hasattr(docling_meta_object, 'origin'):
                                current_chunk_metadata['origin'] = docling_meta_object.origin

                            # 소스 판별 및 메타데이터 추가
                            if hasattr(docling_meta_object, 'table_ref') or 'table' in str(docling_meta_object):
                                current_chunk_metadata["source"] = "table"
                            
                            # 2. 이미지/OCR 소스 확인
                            elif hasattr(docling_meta_object, 'ocr_ref') or 'ocr' in str(docling_meta_object):
                                current_chunk_metadata["source"] = "image"
                                current_chunk_metadata["extraction_method"] = "ocr"
                                
                                # 이미지 설명 추가 (ParsedDocument의 images_descriptions에서 가져옴)
                                if hasattr(docling_meta_object, 'image_id') and parsed_document.image_descriptions:
                                    image_id = docling_meta_object.image_id
                                    for img_desc in parsed_document.image_descriptions:
                                        if img_desc.get('id') == image_id:
                                            current_chunk_metadata["image_description"] = img_desc.get('description', '설명 없음')
                                            break
                                
                            elif hasattr(docling_meta_object, 'image_ref') or 'image' in str(docling_meta_object):
                                current_chunk_metadata["source"] = "image"
                                
                                # 이미지 설명 추가 (ParsedDocument의 images에서 가져옴)
                                if hasattr(docling_meta_object, 'image_id') and parsed_document.images:
                                    image_id = docling_meta_object.image_id
                                    for img in parsed_document.images:
                                        if img.get('id') == image_id:
                                            current_chunk_metadata["image_description"] = img.get('description', '설명 없음')
                                            break
                                
                                # 이미지 설명 확인하여 출력
                                if "image_description" in current_chunk_metadata:
                                    logger.info(f"  이미지 설명: {current_chunk_metadata['image_description']}")
                            
                            # 3. 수식 소스 확인
                            elif hasattr(docling_meta_object, 'equation_ref') or 'formula' in str(docling_meta_object):
                                current_chunk_metadata["source"] = "formula"
                            
                            # 4. 코드 소스 확인
                            elif hasattr(docling_meta_object, 'code_ref') or 'code' in str(docling_meta_object):
                                current_chunk_metadata["source"] = "code"
                            
                            # 5. 기본 텍스트 소스
                            else:
                                current_chunk_metadata["source"] = "text"
                        else:
                            current_chunk_metadata["source"] = "text"  # 기본 소스 값

                        logger.info(f"Processed chunk result {chunk_index} from Docling")

                        if chunk_content:
                            # 제목 정보가 있으면 청크 앞에 추가
                            heading_text = ""
                            if hasattr(docling_meta_object, 'headings') and docling_meta_object.headings:
                                headings = docling_meta_object.headings
                                # 리스트 또는 문자열 형태의 headings 처리
                                if isinstance(headings, list):
                                    # 가장 가까운 제목 최대 2개만 가져오기
                                    relevant_headings = headings[-2:] if len(headings) > 1 else headings
                                    heading_text = " > ".join(relevant_headings)
                                elif isinstance(headings, str):
                                    heading_text = headings
                                
                                if heading_text:
                                    # 제목을 청크 앞에 추가 (구분자 사용)
                                    chunk_content = f"[제목: {heading_text}]\n\n{chunk_content}"
                                    logger.info(f"청크에 제목 추가됨: {heading_text}")
                            
                            chunks.append(DocumentChunk(content=chunk_content, metadata=current_chunk_metadata))
                            chunk_index += 1
                            logger.info(f"Docling 청크 생성: 길이={len(chunk_content)} 문자")
                        else:
                            logger.warning(f"Skipping empty Docling chunk result from library at index {chunk_index}")

                    logger.info("DoclingChunkerAdapter: Docling chunker returned an empty or processed iterator.")
                else:
                    logger.error("DoclingChunkerAdapter: Docling chunker did not return an iterable result.")
                    raise ChunkingError("Docling chunker did not return an iterable result.")

            except Exception as e:
                logger.error(f"DoclingChunkerAdapter: Error during actual Docling chunking - {e}")
                raise ChunkingError(f"Docling failed to chunk: {e}") from e

        else:  # self._chunker 인스턴스가 없거나 내부 객체가 없다면 폴백 로직 실행
            logger.info("DoclingChunkerAdapter: Using fallback simple quantitative chunking (Docling Chunker not available or no internal doc).")
            text = parsed_document.content
            text_len = len(text)
            start = 0
            chunk_index = 0
            base_metadata = parsed_document.metadata.copy()
            base_metadata.pop('__internal_docling_document__', None)

            # 테이블 감지 및 처리 - 폴백 로직에 추가
            # 여기서 파싱된 문서에서 테이블을 찾고 각 테이블에 대해 별도의 청크 생성
            tables_processed = False
            processed_table_hashes = set()  # 중복 테이블 방지를 위한 해시 집합
            
            if hasattr(parsed_document, 'tables') and parsed_document.tables and len(parsed_document.tables) > 0:
                logger.info(f"테이블 감지됨: {len(parsed_document.tables)}개 테이블 개별 처리")
                
                for table_idx, table in enumerate(parsed_document.tables):
                    table_content = ""
                    table_position = None
                    
                    # 테이블 콘텐츠 추출
                    if isinstance(table, dict):
                        if 'content' in table and table['content']:
                            table_content = table['content']
                        if 'position' in table:
                            table_position = table['position']
                        # 테이블 구조 정보가 있으면 추출
                        if 'structure' in table and table['structure']:
                            # 테이블 구조 정보를 문자열로 변환하여 콘텐츠에 추가
                            structure_str = str(table['structure'])
                            # 구조 정보를 테이블 내용에 추가 (구분자 추가)
                            table_content = f"{table_content}\n\n[테이블 구조]\n{structure_str}"
                            logger.info(f"테이블 {table_idx+1}에 구조 정보 추가됨 (길이: {len(structure_str)} 문자)")
                    
                    # 테이블 콘텐츠가 없는 경우는 건너뜀
                    if not table_content:
                        logger.info(f"테이블 {table_idx+1}의 내용이 없어 처리하지 않음")
                        continue
                    
                    # 테이블 콘텐츠 해시 생성 (중복 방지)
                    import hashlib
                    table_hash = hashlib.md5(table_content.encode('utf-8')).hexdigest()
                    
                    # 이미 처리한 동일 테이블인지 확인
                    if table_hash in processed_table_hashes:
                        logger.info(f"테이블 {table_idx+1}은 이미 처리된 중복 테이블이므로 건너뜀")
                        continue
                    
                    # 해시를 추가하여 이후 중복 방지
                    processed_table_hashes.add(table_hash)
                    
                    # 테이블 메타데이터 생성
                    table_metadata = base_metadata.copy()
                    table_metadata["chunk_index"] = chunk_index
                    table_metadata["source"] = "table"  # 소스를 'table'로 명시적 설정
                    table_metadata["table_index"] = table_idx
                    table_metadata["table_hash"] = table_hash  # 테이블 해시 추가
                    
                    # 테이블 콘텐츠 정보 로깅
                    content_preview = table_content[:100] + "..." if len(table_content) > 100 else table_content
                    logger.info(f"테이블 {table_idx+1} 내용: {content_preview}")
                    
                    if table_position:
                        table_metadata["position"] = table_position
                        
                    # 테이블 구조 정보 추가
                    if 'structure' in table:
                        table_metadata["table_structure"] = str(table['structure'])[:500]  # 구조 정보 일부만 저장
                        # 로그에 테이블 구조 정보 추가됨을 표시
                        logger.info(f"  - 메타데이터 테이블 구조: 있음 (메타데이터에 일부 저장)")
                    
                    # 테이블은 청킹하지 않고 하나의 청크로 저장
                    chunks.append(DocumentChunk(content=table_content, metadata=table_metadata))
                    
                    # 테이블 청크 생성 후 로그 메시지 개선
                    logger.info(f"테이블 {table_idx+1} 청크 생성 완료:")
                    logger.info(f"  - 내용 길이: {len(table_content)} 문자")
                    logger.info(f"  - 메타데이터 소스: {table_metadata.get('source')}")
                    logger.info(f"  - 테이블 인덱스: {table_metadata.get('table_index')}")
                    logger.info(f"  - 테이블 해시: {table_metadata.get('table_hash')[:8]}...")
                    
                    chunk_index += 1
                    tables_processed = True
            
            # 추가: 청크 객체 유효성 검사 (테이블 청크 포함)
            if chunks:
                for i, chunk in enumerate(chunks):
                    source_type = chunk.metadata.get('source', 'unknown')
                    logger.info(f"청크 {i} 최종 검사: 타입={source_type}, 길이={len(chunk.content)}")
            
            # 일반 텍스트 청킹 항상 수행 - 기존의 if-else 구조 제거
            logger.info("일반 텍스트 청킹 진행")
            
            while start < text_len:
                end = min(start + self._chunk_size, text_len)
                
                # 청크 경계 조정 개선 (문장 또는 단락 끝에서 자르기)
                if end < text_len:
                    # 우선 큰 단위(단락) 먼저 검색 - 더 멀리 검색 범위 확장
                    next_para = text.find('\n\n', start, min(end + 500, text_len))
                    
                    # 단락 구분이 발견되면 그 위치로 끝 지점 조정
                    if next_para > start and next_para < end + 500:
                        end = next_para + 2
                    else:
                        # 단락이 없으면 2순위로 문장 끝 찾기
                        # 마침표+공백 다음에 대문자가 오는 패턴 찾기
                        next_sentence_end = -1
                        for match in re.finditer(r'\. [A-Z가-힣]', text[start:min(end + 200, text_len)]):
                            potential_end = start + match.start() + 1  # 마침표 위치까지만
                            if potential_end > start and potential_end > end - 300:  # 너무 짧은 청크 방지
                                next_sentence_end = potential_end
                                break
                        
                        # 문장 끝이 발견되면 조정
                        if next_sentence_end > 0:
                            end = next_sentence_end + 1
                        else:
                            # 줄바꿈 찾기 (보다 멀리 검색)
                            next_newline = text.find('\n', start, min(end + 200, text_len))
                            if next_newline > start and next_newline < end + 200:
                                end = next_newline + 1
                
                chunk_content = text[start:end].strip()
                
                # 내용이 있고 최소 길이(100자) 이상인 경우에만 청크 생성
                # 너무 짧은 청크는 버림
                if chunk_content and len(chunk_content) >= 100:
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata["chunk_index"] = chunk_index
                    chunk_metadata["start_char"] = start
                    chunk_metadata["end_char"] = end
                    chunk_metadata["source"] = "text"  # 명시적으로 'text' 지정
                    
                    # 제목 정보 추가
                    chunk_content = self._add_heading_to_content(chunk_content, base_metadata)
                    
                    chunks.append(DocumentChunk(content=chunk_content, metadata=chunk_metadata))
                    chunk_index += 1
                    logger.info(f"일반 텍스트 청크 생성: 길이={len(chunk_content)} 문자")
                else:
                    logger.info(f"청크가 너무 짧아 건너뜀: 길이={len(chunk_content)} 문자")
                
                # 다음 청크 시작 위치로 이동
                start = end
            
            # 테이블이 처리되었다는 메시지는 유지하되, 텍스트 청킹은 항상 실행
            if tables_processed:
                logger.info("테이블 데이터에 대한 청크도 생성되었습니다")
                
            # 이미지 설명 정보 처리 부분 유지 (이전과 동일)
            if hasattr(parsed_document, 'image_descriptions') and parsed_document.image_descriptions:
                logger.info(f"[CHUNKER] {len(parsed_document.image_descriptions)}개 이미지 설명 청킹 처리")
                
                for img_idx, img_desc in enumerate(parsed_document.image_descriptions):
                    # 이미지 설명 추출
                    description = ""
                    image_id = ""
                    
                    if isinstance(img_desc, dict):
                        description = img_desc.get('description', '')
                        image_id = img_desc.get('image_id', '')
                    else:
                        # 객체인 경우 속성 접근 시도
                        description = getattr(img_desc, 'description', '')
                        image_id = getattr(img_desc, 'image_id', '')
                    
                    # 설명이 없으면 건너뜀
                    if not description or not isinstance(description, str) or len(description.strip()) == 0:
                        continue
                    
                    # 이미지 청크 메타데이터
                    img_metadata = base_metadata.copy()
                    img_metadata["chunk_index"] = chunk_index
                    img_metadata["source"] = "image"
                    img_metadata["image_id"] = image_id
                    
                    # 이미지 설명을 콘텐츠로 사용하되, 이미지 식별자 추가
                    img_content = f"[이미지 {img_idx+1} 설명]\n{description}"
                    
                    # 이미지 청크 추가
                    chunks.append(DocumentChunk(content=img_content, metadata=img_metadata))
                    logger.info(f"이미지 설명 청크 추가: {img_idx+1}/{len(parsed_document.image_descriptions)}")
                    chunk_index += 1
            
            logger.info(f"DoclingChunkerAdapter: Chunking process finished. Generated {len(chunks)} chunks.")

        # 청크 생성 결과 출력 (일부만)
        logger.info(f"[CHUNKING] 성공: {len(chunks)}개 청크 생성됨")
        # 미리보기 (첫 3개 청크)
        display_limit = min(3, len(chunks))
        for i in range(display_limit):
            chunk = chunks[i]
            logger.info(f"  청크 {i+1}: {len(chunk.content)} 문자, 소스={chunk.metadata.get('source', '알 수 없음')}")
            
            # 이미지 설명 정보 출력 (있는 경우)
            if 'image_description' in chunk.metadata and chunk.metadata['image_description']:
                desc_preview = str(chunk.metadata['image_description'])[:100]
                if len(str(chunk.metadata['image_description'])) > 100:
                    desc_preview += "..."
                logger.info(f"  이미지 설명: {desc_preview}")

        # 나머지 청크 수만 출력
        if len(chunks) > display_limit:
            logger.info(f"  ... 외 {len(chunks)-display_limit}개 청크")

        return chunks

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        logger.info(f"[CHUNKING] 시작: 입력 텍스트 길이 {len(text)} 문자")
        
        # 기존 코드...
        
        # 청킹 성공 시
        logger.info(f"[CHUNKING] 성공: {len(chunks)}개 청크 생성됨")
        chunk_count = len(chunks)
        for i in range(min(3, chunk_count)):  # 처음 3개만 표시
            if i < chunk_count:  # 안전 검사
                chunk = chunks[i]
                logger.info(f"  청크 {i+1}: {len(chunk.content)} 문자")
            # 청크가 이미지 소스인 경우 설명 추가 출력
            if chunk.metadata.get("source") == "image" and "image_description" in chunk.metadata:
                    img_desc = chunk.metadata.get('image_description', '')
                    desc_preview = ""
                    if isinstance(img_desc, str):
                        if len(img_desc) > 50:
                            desc_preview = img_desc[0:50] + "..."
                        else:
                            desc_preview = img_desc
                    else:
                        desc_preview = str(img_desc)
                    logger.info(f"  이미지 설명: {desc_preview}")
        if chunk_count > 3:
            logger.info(f"  ... 외 {chunk_count-3}개 청크")
        
        return chunks

    def _debug_parsed_content(self, parsed_document):
        """입력 데이터 디버깅용 함수. 실제 청킹 전 파싱된 문서 내용 확인."""
        try:
            # 청킹 단계 입력 데이터 확인 (디버깅용)
            logger.info("\n----- 청킹 단계 입력 데이터 확인 -----")
            logger.info(f"[CHUNKS] 원본 텍스트 길이: {len(parsed_document.content)} 글자")
            
            # 테이블 정보 확인
            table_count = 0
            if hasattr(parsed_document, 'tables') and parsed_document.tables:
                table_count = len(parsed_document.tables)
                logger.info(f"[CHUNKS] 테이블 수: {table_count}")
                
                # 테이블 정보 샘플 (처음 2개만)
                display_limit = min(2, table_count)
                for i in range(display_limit):
                    table = parsed_document.tables[i]
                    # 테이블 구조 문자열 생성
                    structure_str = "정보 없음"
                    if isinstance(table, dict) and 'structure' in table:
                        structure_str = str(table['structure'])[:100] + "..." if len(str(table['structure'])) > 100 else str(table['structure'])
                    logger.info(f"  - 테이블 {i+1} 구조: {structure_str}")
            
            # 이미지 정보 확인
            image_count = 0
            if hasattr(parsed_document, 'images') and parsed_document.images:
                image_count = len(parsed_document.images)
                logger.info(f"[CHUNKS] 이미지 수: {image_count}")
                
                # 이미지 정보 샘플 (처음 2개만)
                display_limit = min(2, image_count)
                for i in range(display_limit):
                    img = parsed_document.images[i]
                    # 이미지 설명 문자열 생성
                    description_str = "정보 없음"
                    if isinstance(img, dict) and 'description' in img:
                        description_str = str(img['description'])[:100] + "..." if len(str(img['description'])) > 100 else str(img['description'])
                    logger.info(f"  - 이미지 {i+1} 설명: {description_str}")
            
            # 이미지 설명 정보 확인
            desc_count = 0
            if hasattr(parsed_document, 'image_descriptions') and parsed_document.image_descriptions:
                desc_count = len(parsed_document.image_descriptions)
                logger.info(f"[CHUNKS] 이미지 설명 정보 수: {desc_count}")
                
                # 이미지 설명 정보 샘플 (처음 2개만)
                display_limit = min(2, desc_count)
                for i in range(display_limit):
                    desc = parsed_document.image_descriptions[i]
                    # id와 설명 문자열 생성
                    img_id = "알 수 없음"
                    desc_preview = "정보 없음"
                    
                    if hasattr(desc, 'id'):
                        img_id = str(desc.id)
                    
                    if hasattr(desc, 'text'):
                        desc_text = desc.text
                        desc_preview = str(desc_text)[:100] + "..." if len(str(desc_text)) > 100 else str(desc_text)
                    
                    logger.info(f"  - 설명 {i+1}: ID={img_id}, 내용={desc_preview}")
            
            # 원본 Docling 문서 객체 확인 (있는 경우)
            if hasattr(parsed_document, 'docling_document') and parsed_document.docling_document:
                logger.info("[CHUNKS] Docling 내부 문서 객체 있음")
                doc = parsed_document.docling_document
                
                # 레이아웃 정보 확인
                if hasattr(doc, 'layout') and doc.layout:
                    logger.info(f"  - 레이아웃 정보 있음: {type(doc.layout).__name__}")
                    # 레이아웃 항목 수
                    if hasattr(doc.layout, 'items'):
                        logger.info(f"  - 레이아웃 항목 수: {len(doc.layout.items) if isinstance(doc.layout.items, list) else 'N/A'}")
            else:
                logger.info("[CHUNKS] Docling 내부 문서 객체 없음")
            
            logger.info("----- 청킹 단계 입력 데이터 확인 종료 -----\n")
        
        except Exception as e:
            logger.error(f"청킹 데이터 디버깅 오류: {e}")

    # 단순 청킹 메서드 (폴백) - 라인 단위로 나누기
    def _simple_text_chunking(self, text: str) -> List[DocumentChunk]:
        """
        텍스트를 단순 텍스트 분할 방식으로 청킹합니다. (폴백 메서드)
        한 라인이 chunk_size를 초과하면 여러 청크로 나눕니다.
        빈 라인은 개별 청크로 취급하지 않습니다.
        """
        # 청킹 시작 로깅
        logger.info(f"[CHUNKING] 시작: 입력 텍스트 길이 {len(text)} 문자")

        chunks: List[DocumentChunk] = []
        lines = text.split('\n')
        
        # 각 라인 처리
        current_chunk_lines = []
        current_length = 0
        
        for line in lines:
            line_length = len(line)
            
            # 현재 라인이 청크 크기를 초과하면 여러 청크로 분할
            if line_length > self._chunk_size:
                # 기존에 모인 내용이 있으면 청크로 만들기
                if current_chunk_lines and current_length > 0:
                    chunk_text = '\n'.join(current_chunk_lines)
                    chunks.append(DocumentChunk(content=chunk_text, metadata={}))
                    current_chunk_lines = []
                    current_length = 0
                
                # 긴 라인 분할
                words = line.split(' ')
                temp_chunk = ""
                
                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= self._chunk_size:
                        if temp_chunk:
                            temp_chunk += ' ' + word
                        else:
                            temp_chunk = word
                    else:
                        if temp_chunk:
                            chunks.append(DocumentChunk(content=temp_chunk, metadata={}))
                        temp_chunk = word
                
                # 마지막 temp_chunk 추가
                if temp_chunk:
                    chunks.append(DocumentChunk(content=temp_chunk, metadata={}))
            
            # 현재 라인 추가시 청크 크기 초과하는 경우
            elif current_length + line_length + 1 > self._chunk_size:
                # 기존 내용으로 청크 생성
                chunk_text = '\n'.join(current_chunk_lines)
                chunks.append(DocumentChunk(content=chunk_text, metadata={}))
                
                # 새 청크 시작
                current_chunk_lines = [line]
                current_length = line_length
            
            # 현재 청크에 라인 추가
            else:
                current_chunk_lines.append(line)
                current_length += line_length + 1  # +1 for newline
        
        # 마지막 청크 처리
        if current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines)
            chunks.append(DocumentChunk(content=chunk_text, metadata={}))
        
        # 청크 생성 결과 출력 (일부만)
        logger.info(f"[CHUNKING] 성공: {len(chunks)}개 청크 생성됨")
        # 미리보기 (첫 3개 청크)
        chunk_count = len(chunks)
        display_limit = min(3, chunk_count)
        for i in range(display_limit):
            chunk = chunks[i]
            logger.info(f"  청크 {i+1}: {len(chunk.content)} 문자")
            
            # 이미지 설명 정보 출력 (있는 경우)
            if 'image_description' in chunk.metadata and chunk.metadata['image_description']:
                desc_preview = str(chunk.metadata['image_description'])[:100]
                if len(str(chunk.metadata['image_description'])) > 100:
                    desc_preview += "..."
                logger.info(f"  이미지 설명: {desc_preview}")

        # 나머지 청크 수만 출력
        if chunk_count > display_limit:
            logger.info(f"  ... 외 {chunk_count-display_limit}개 청크")
        
        return chunks

    # 청크 내용에 제목 추가하는 유틸리티 함수
    def _add_heading_to_content(self, chunk_content: str, metadata: Dict[str, Any]) -> str:
        """청크 내용에 제목 정보가 있으면 추가하는 함수"""
        if not chunk_content:
            return chunk_content
            
        heading_text = ""
        if 'headings' in metadata and metadata['headings']:
            headings = metadata['headings']
            # 리스트 또는 문자열 형태의 headings 처리
            if isinstance(headings, list):
                # 가장 가까운 제목 최대 2개만 가져오기
                relevant_headings = headings[-2:] if len(headings) > 1 else headings
                heading_text = " > ".join(relevant_headings)
            elif isinstance(headings, str):
                heading_text = headings
            
            if heading_text:
                # 제목을 청크 앞에 추가 (구분자 사용)
                chunk_content = f"[제목: {heading_text}]\n\n{chunk_content}"
                logger.info(f"청크에 제목 추가됨: {heading_text}")
                
        return chunk_content

    # 청크 크기와 오버랩 가져오기 메서드 - 설정 파일에서 최신 값을 읽거나 인스턴스 값 반환
    def get_chunk_size(self) -> int:
        """현재 청크 크기 반환 (settings 업데이트 고려)"""
        # 런타임에 config 변경사항 반영을 원한다면 이 부분을 수정
        # return self._settings.CHUNK_SIZE  # 항상 최신 설정값 사용
        return self._chunk_size  # 인스턴스 생성 시 설정된 값 사용
        
    def get_chunk_overlap(self) -> int:
        """현재 청크 오버랩 크기 반환 (settings 업데이트 고려)"""
        # 런타임에 config 변경사항 반영을 원한다면 이 부분을 수정
        # return self._settings.CHUNK_OVERLAP  # 항상 최신 설정값 사용
        return self._chunk_overlap  # 인스턴스 생성 시 설정된 값 사용

    # 청크 크기와 오버랩 설정 메서드
    def set_chunk_size(self, size: int) -> None:
        """청크 크기 설정"""
        if size <= 0:
            raise ValueError("Chunk size must be positive")
        self._chunk_size = size
        logger.info(f"청크 크기가 {size}(으)로 변경되었습니다.")
        
    def set_chunk_overlap(self, overlap: int) -> None:
        """청크 오버랩 크기 설정"""
        if overlap < 0:
            raise ValueError("Chunk overlap must be non-negative")
        if overlap >= self._chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        self._chunk_overlap = overlap
        logger.info(f"청크 오버랩이 {overlap}(으)로 변경되었습니다.")