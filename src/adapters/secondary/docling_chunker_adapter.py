# src/adapters/secondary/docling_chunker_adapter.py

import logging
from typing import List, Dict, Any, Iterable, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

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

        # --- 폴백 청킹 설정 파라미터 이름을 main.py 호출과 일치하도록 수정 ---
        chunk_size: int = 1000,  # <-- 이름이 chunk_size 로 변경되었습니다!
        chunk_overlap: int = 200,  # <-- 이름이 chunk_overlap 로 변경되었습니다!
    ):
        """
        DoclingChunkerAdapter 초기화. Docling HybridChunker 인스턴스를 생성하거나 폴백 설정을 가집니다.

        Args:
            tokenizer: Docling HybridChunker에 사용될 토크나이저 설정.
            max_tokens: Docling HybridChunker의 최대 토큰 수 설정.
            merge_peers: Docling HybridChunker의 merge_peers 설정.
            # 기타 Docling HybridChunker 초기화 파라미터들 (Docling 문서 확인 필요)
            chunk_size: Docling Chunker 또는 폴백 청킹 크기. # <-- Docstring 업데이트
            chunk_overlap: Docling Chunker 또는 폴백 청킹 오버랩. # <-- Docstring 업데이트
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size.")

        self._chunker: Optional[HybridChunker] = None
        # --- 인스턴스 변수 이름도 통일 ---
        self._chunk_size = chunk_size  # <-- 인스턴스 변수 이름도 _chunk_size로 변경
        self._chunk_overlap = chunk_overlap  # <-- 인스턴스 변수 이름도 _chunk_overlap로 변경

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
                            chunks.append(DocumentChunk(content=chunk_content, metadata=current_chunk_metadata))
                            chunk_index += 1
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

            while start < text_len:
                end = min(start + self._chunk_size, text_len)
                chunk_content = text[start:end]

                chunk_metadata = base_metadata.copy()
                chunk_metadata["chunk_index"] = chunk_index
                chunk_metadata["start_char"] = start
                chunk_metadata["end_char"] = end

                chunks.append(DocumentChunk(content=chunk_content, metadata=chunk_metadata))

                if end == text_len:
                    break
                next_start = start + (self._chunk_size - self._chunk_overlap)
                start = max(start + 1, next_start)
                if start >= text_len:
                    break
                chunk_index += 1

            logger.info(f"DoclingChunkerAdapter: Chunking process finished. Generated {len(chunks)} chunks.")

        # 청크 생성 결과 출력 (일부만)
        logger.info(f"[CHUNKING] 성공: {len(chunks)}개 청크 생성됨")
        # 미리보기 (첫 3개 청크)
        display_limit = min(3, len(chunks))
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