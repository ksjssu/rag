# src/adapters/secondary/docling_chunker_adapter.py

# --- Docling 라이브러리 임포트 ---
# 실제 사용하시는 Docling 라이브러리의 정확한 임포트 구문을 사용하세요.
# 제공해주신 HybridChunker 코드 기반으로 추정한 임포트입니다.
try:
    # Docling 청킹을 위한 핵심 클래스 임포트
    from docling_core.transforms.chunker import HybridChunker # <-- HybridChunker 클래스 임포트
    # Docling 내부 문서 객체 타입 정의가 필요할 수 있으나,
    # 파서 어댑터에서 Any로 넘겨주므로 여기서는 명시적 임포트 없이 Any로 처리합니다.
    # 예: from docling_core.datamodel.document import Document as DoclingDocument # 충돌 방지 이름 변경

    # HybridChunker가 의존하는 내부 클래스 임포트 (초기화 파라미터 타입 힌트용)
    from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer # Tokenizer 타입
    # 필요한 다른 Docling 클래스 임포트 (DocChunk, DocMeta, BaseChunk 등 - 결과 처리 시 사용)
    # from docling_core.transforms.chunker import DocChunk, DocMeta, BaseChunk

    _docling_chunker_available = True
    print("Docling chunking library (HybridChunker) imported successfully.")
    # Docling Chunker 인스턴스는 초기화 시점에 생성 (__init__에서 수행)
except ImportError:
    print("Warning: Docling HybridChunker or dependencies not found (`docling_core.transforms`, `semchunk`). DoclingChunkerAdapter will use simple line splitting.")
    _docling_chunker_available = False
    # --- Docling 클래스가 없을 경우 에러 방지를 위한 더미 클래스 정의 ---
    # 실제 HybridChunker 클래스의 시그니처와 최대한 유사하게 정의하여 타입 힌트 에러를 줄입니다.
    # 제공된 HybridChunker 코드 기반으로 더미 클래스 업데이트
    print("   Using dummy Docling chunker classes.")
    class HybridChunker: # <-- HybridChunker 더미 클래스 정의
        def __init__(self, tokenizer="dummy_tokenizer", max_tokens=1000, merge_peers=True, **kwargs): # 실제 생성자 시그니처 반영 시도
             print(f"   (Simulating HybridChunker initialization - Library not available) tokenizer={tokenizer}, max_tokens={max_tokens}, merge_peers={merge_peers}")
             self._tokenizer = tokenizer # 더미 클래스에서도 설정 저장 예시
             self._max_tokens = max_tokens
             self._merge_peers = merge_peers
             # 다른 초기화 파라미터도 받도록 **kwargs 추가

        def chunk(self, doc, **kwargs): # chunk 메서드 시그니처 예시 반영
            print("   (Simulating HybridChunker.chunk - Library not available)")
            # 더미 청크 결과 반환 (실제 청크 객체와 유사하게 .text 속성을 가짐)
            # 입력 doc가 ParsedDocument 객체이고 '__internal_docling_document__'가 None인 경우를 시뮬레이션
            text_content = ""
            if hasattr(doc, 'content') and isinstance(doc.content, str):
                 text_content = doc.content # ParsedDocument의 텍스트 사용 가정 (fallback 시)
            elif doc is not None and hasattr(doc, 'export_to_markdown'): # 내부 Docling 객체가 있다면 (실제 라이브러리에서)
                 # Docling 내부 객체에서 텍스트 가져오는 시뮬레이션
                 try: text_content = doc.export_to_markdown() # export_to_markdown 사용 시도
                 except: pass
                 if not text_content and hasattr(doc, 'text'): text_content = doc.text # .text 사용 시도
                 if not text_content and hasattr(doc, 'get_text'): text_content = doc.get_text() # get_text 사용 시도


            # シンプルな行分割で代替
            class MockDocChunkResult: # Docling chunk result 객체 흉내 (DocChunk 기반 추정)
                 def __init__(self, text, meta=None): self.text = text; self.meta = meta or {} # .meta 속성 추가
                 @property # 속성으로 정의
                 def text(self): return self._text
                 @text.setter
                 def text(self, value): self._text = value
                 @property # 속성으로 정의
                 def meta(self): return self._meta or {}
                 @meta.setter
                 def meta(self, value): self._meta = value
                 # DocMeta 객체의 속성 흉내 (Docling 문서 확인)
                 @property
                 def headings(self): return self.meta.get('headings')
                 @property
                 def captions(self): return self.meta.get('captions')
                 @property
                 def origin(self): return self.meta.get('origin') # 원본 위치 정보
                 # 기타 필요한 속성 추가


            # 입력 텍스트를 줄바꿈 기준으로 나누어 더미 청크 결과 목록 생성
            return [MockDocChunkResult(line.strip(), {"source_line": i+1}) for i, line in enumerate(text_content.split('\n')) if line.strip()]


    # --- 어댑터 특정 예외 정의 ---
    # 청킹 과정에서 발생하는 오류를 나타내기 위한 어댑터 레벨의 예외
    class ChunkingError(Exception):
        """Represents an error during the text chunking process."""
        pass


from ports.output_ports import TextChunkingPort # 구현할 포트 임포트
from domain.models import ParsedDocument, DocumentChunk # 입/출력 도메인 모델 임포트
from typing import List, Dict, Any, Iterable, Optional, Union # Iterable, Optional, Union 임포트


class DoclingChunkerAdapter(TextChunkingPort):
    """
    Docling 라이브러리(HybridChunker)를 사용하여 TextChunkingPort를 구현하는 어댑터.
    ParsedDocument의 메타데이터에서 Docling 내부 문서 객체를 추출하여 청킹합니다.
    """
    def __init__(
        self,
        # HybridChunker 초기화에 필요한 파라미터들을 __init__에서 받습니다.
        # 제공된 HybridChunker 코드의 Pydantic 필드를 기반으로 파라미터 정의
        tokenizer: Union[str, BaseTokenizer] = "sentence-transformers/all-MiniLM-L6-v2", # 토크나이저 설정 (이름 또는 인스턴스)
        max_tokens: Optional[int] = None, # 최대 토큰 수 (None이면 토크나이저 기본값)
        merge_peers: bool = True, # undersized chunks 병합 여부
        # 기타 HybridChunker 초기화 파라미터가 있다면 여기에 추가 (Docling 문서 확인)
        # 예: splitting_strategies: List[str] = ['recursive', 'by_title']

        # Docling 없을 때 폴백 청킹 설정은 그대로 유지
        fallback_chunk_size: int = 1000,
        fallback_chunk_overlap: int = 200,
    ):
        """
        DoclingChunkerAdapter 초기화. Docling HybridChunker 인스턴스를 생성하거나 폴백 설정을 가집니다.

        Args:
            tokenizer: Docling HybridChunker에 사용될 토크나이저 설정 (모델 이름 또는 BaseTokenizer 인스턴스).
            max_tokens: Docling HybridChunker의 최대 토큰 수 설정. None이면 토크나이저 기본값 사용.
            merge_peers: Docling HybridChunker의 merge_peers 설정.
            # 기타 Docling HybridChunker 초기화 파라미터들 (Docling 문서 확인 필요)
            fallback_chunk_size: Docling 없을 때 폴백 청킹 크기.
            fallback_chunk_overlap: Docling 없을 때 폴백 청킹 오버랩.
        """
        if fallback_chunk_overlap >= fallback_chunk_size:
             raise ValueError("Fallback chunk overlap must be less than fallback chunk size.")

        self._chunker: Optional[HybridChunker] = None # Docling HybridChunker 인스턴스
        self._fallback_chunk_size = fallback_chunk_size # Docling 없을 때 폴백 설정
        self._fallback_chunk_overlap = fallback_chunk_overlap

        # Docling HybridChunker 초기화에 필요한 파라미터들을 저장
        self._chunker_init_params: Dict[str, Any] = {
            'tokenizer': tokenizer,
            'max_tokens': max_tokens, # max_tokens는 Pydantic 필드로 존재 (None 허용)
            'merge_peers': merge_peers,
            # 예: 'splitting_strategies': splitting_strategies,
            # 기타 HybridChunker 생성자가 받는 파라미터 추가 (Docling 문서 확인)
        }


        if _docling_chunker_available:
            print("DoclingChunkerAdapter: Initializing Docling HybridChunker...")
            try:
                # --- Docling HybridChunker 인스턴스 생성 ★ 실제 초기화 ★ ---
                # 제공된 HybridChunker 코드의 __init__ (Pydantic BaseModel)에 정의된 파라미터들을 전달
                self._chunker = HybridChunker(
                    **self._chunker_init_params # 저장된 파라미터 딕셔너리 언팩하여 전달
                )
                print("DoclingChunkerAdapter: Docling HybridChunker initialized successfully.")
            except Exception as e: # Docling Chunker 초기화 중 발생할 수 있는 예외 처리
                 print(f"Error initializing Docling HybridChunker: {e}")
                 self._chunker = None # 초기화 실패 시 None
                 # 초기화 실패 시 ChunkingError 예외를 발생시켜 앱 시작 중단 고려
                 # raise ChunkingError(f"Failed to initialize Docling HybridChunker: {e}") from e


        if self._chunker is None:
             print(f"DoclingChunkerAdapter: Docling HybridChunker not available or failed to initialize. Will use simple quantitative chunking (size={self._fallback_chunk_size}, overlap={self._fallback_chunker_overlap}).")


    def chunk(self, parsed_document: ParsedDocument) -> List[DocumentChunk]:
        """
        ParsedDocument를 Docling HybridChunker로 청킹합니다.
        """
        print(f"DoclingChunkerAdapter: Starting chunking process...")

        chunks: List[DocumentChunk] = []
        # 파서 어댑터에서 ParsedDocument 메타데이터에 담아 전달한 Docling 내부 문서 객체를 추출
        # '__internal_docling_document__' 키는 파서 어댑터 코드와 일치해야 합니다.
        docling_internal_document = parsed_document.metadata.get('__internal_docling_document__')

        # --- Docling 청킹 로직 또는 폴백 로직 실행 ---

        if self._chunker and docling_internal_document: # Docling 청커 인스턴스가 유효하고 내부 객체도 있다면 실제 청킹 실행
            print("DoclingChunkerAdapter: Using configured Docling HybridChunker.")
            try:
                # Docling 내부 문서 객체가 HybridChunker.chunk 메서드의 입력 타입 (DoclingDocument)과
                # 호환되는지 확인 (파서 어댑터가 올바른 객체를 넘겨줬다고 가정)
                # if not isinstance(docling_internal_document, DoclingDocument): # DoclingDocument 임포트 가능 시 타입 확인
                #     raise ChunkingError("Internal Docling document object is of incorrect type.")

                # --- ★★★ 실제 Docling 라이브러리 청킹 기능을 호출하는 부분 ★★★ ---
                print("   Calling self._chunker.chunk()...")
                # self._chunker는 __init__에서 생성한 HybridChunker 인스턴스
                # .chunk() 메서드는 Docling 라이브러리의 핵심 청킹 메서드
                # 입력은 파서 결과로 얻은 Docling 내부 문서 객체 (docling_internal_document)
                # 청킹 옵션 (크기, 오버랩, 전략 등)이 메서드 호출 시 파라미터로 전달될 수 있습니다 (Docling 문서 확인)
                # 제공된 HybridChunker 코드의 chunk 메서드는 **kwargs만 받으므로, 옵션은 __init__에서 설정하는 것이 일반적일 것 같습니다.**
                # 하지만 혹시 chunk 메서드에 옵션 전달이 가능하다면 여기에 추가합니다.
                docling_chunk_results_iterator: Iterable = self._chunker.chunk( # <--- ▶︎▶︎▶︎ 실제 호출 라인! ◀︎◀︎◀︎
                    docling_internal_document,
                    # Docling 문서에서 chunk 메서드가 받는 파라미터 확인 (예: chunk_size, overlap 등)
                    # chunk_size=self._fallback_chunk_size, # 예시: chunk 메서드에 옵션 전달 (HybridChunker가 지원 시)
                    # overlap=self._fallback_chunk_overlap, # 예시
                    # **kwargs # chunk 메서드가 **kwargs를 받는다면
                )
                # --- 호출 결과는 청크 객체들의 이터레이터입니다 (Docling 문서에 정의). ---
                print("   Received Docling chunking results iterator.")


                # --- Docling 청킹 결과 목록을 순회하며 처리 ---
                print("   Processing Docling chunking results iterator...")
                # Docling 청킹 결과 목록의 각 요소(DocChunk 타입)에서 청크 내용과 메타데이터를 추출하여
                # 우리의 DocumentChunk 도메인 모델로 변환합니다. (Docling 문서에서 결과 객체 구조 확인 필요)

                base_metadata = parsed_document.metadata.copy()
                # 내부 Docling 객체는 청크 메타데이터에 포함시키지 않습니다.
                base_metadata.pop('__internal_docling_document__', None)

                chunk_index = 0
                # 결과가 실제로 이터러블한지 확인하고 순회
                if isinstance(docling_chunk_results_iterator, Iterable):
                    for docling_chunk_result in docling_chunk_results_iterator:
                       # ---> Docling 청킹 결과 각 요소에서 데이터 추출 (Docling 문서 확인 필요) ---
                       # 제공된 HybridChunker 코드를 보면 결과 객체는 DocChunk이고 .text 속성과 .meta 속성을 가집니다.
                       # .text는 청크 내용, .meta는 DocMeta 객체입니다.

                       chunk_content = ""
                       if hasattr(docling_chunk_result, 'text') and isinstance(docling_chunk_result.text, str):
                            chunk_content = docling_chunk_result.text
                       elif hasattr(docling_chunk_result, 'get_text'): # 혹시 get_text 메서드도 있다면
                             chunk_content = docling_chunk_result.get_text() or ""


                       # 청크별 메타데이터 추출 (Docling DocChunk의 .meta 속성에서 가져옴)
                       # .meta 속성은 DocMeta 객체이며, .headings, .captions, .origin 등을 가집니다.
                       # 어떤 메타데이터를 우리의 DocumentChunk에 포함시킬지 결정합니다.
                       current_chunk_metadata = base_metadata.copy() # 원본/파싱 메타데이터 복사
                       current_chunk_metadata["chunk_index"] = chunk_index # 우리 시스템의 청크 순서 인덱스

                       docling_meta_object = None
                       if hasattr(docling_chunk_result, 'meta'): # .meta 속성 존재 확인
                            docling_meta_object = docling_chunk_result.meta

                       if docling_meta_object:
                           # DocMeta 객체의 속성에서 필요한 메타데이터 추출 (Docling 문서 확인)
                           # 예: 제목 목록, 캡션 목록, 원본 위치 정보 등
                           if hasattr(docling_meta_object, 'headings'): current_chunk_metadata['headings'] = docling_meta_object.headings
                           if hasattr(docling_meta_object, 'captions'): current_chunk_metadata['captions'] = docling_meta_object.captions
                           if hasattr(docling_meta_object, 'origin'): current_chunk_metadata['origin'] = docling_meta_object.origin # 위치 정보

                           # DocMeta 객체 자체를 포함시킬 수도 있으나, 필요한 정보만 추출하는 것이 좋습니다.
                           # current_chunk_metadata['docling_meta'] = docling_meta_object # 예시: DocMeta 객체 자체 포함


                       print(f"   (Processed chunk result {chunk_index} from Docling)") # 로깅 추가

                       # --------------------------------------------------------------------------


                       if chunk_content: # 내용이 있는 청크만 DocumentChunk 객체로 만들어 추가
                            chunks.append(DocumentChunk(content=chunk_content, metadata=current_chunk_metadata))
                            chunk_index += 1
                       else:
                            # Docling에서 빈 청크 결과가 나올 수도 있습니다. 필요에 따라 로깅 등 처리
                            print(f"   (Skipping empty Docling chunk result from library at index {chunk_index})")


                    else:
                         print("DoclingChunkerAdapter: Docling chunker returned an empty or processed iterator.") # 순회 완료 로그
                         # 결과가 이터러블했지만 내용이 없거나 빈 목록인 경우


                else:
                     print("DoclingChunkerAdapter: Docling chunker did not return an iterable result.")
                     # 결과가 이터러블하지 않은 예상치 못한 상황 처리 (오류 로깅 등)
                     # 이 경우 청킹 실패로 간주하고 예외를 발생시킬 수 있습니다.
                     raise ChunkingError("Docling chunker did not return an iterable result.")

            except Exception as e: # Docling 청킹 중 발생한 예외 처리
                 print(f"DoclingChunkerAdapter: Error during actual Docling chunking - {e}")
                 # Docling 청킹 중 발생한 예외를 어댑터 레벨의 ChunkingError로 변환하여 다시 발생
                 raise ChunkingError(f"Docling failed to chunk: {e}") from e


        else: # self._chunker 인스턴스가 없거나 내부 객체가 없다면 폴백 로직 실행
            print("DoclingChunkerAdapter: Using fallback simple quantitative chunking (Docling Chunker not available or no internal doc).")
            # Docling 내부 객체가 없거나 Docling 청커를 사용할 수 없을 때의 폴백 (심플 정량적 청킹)
            text = parsed_document.content
            text_len = len(text)
            start = 0
            chunk_index = 0
            base_metadata = parsed_document.metadata.copy()
            base_metadata.pop('__internal_docling_document__', None) # 내부 객체 제거

            while start < text_len:
                end = min(start + self._fallback_chunk_size, text_len)
                chunk_content = text[start:end]

                chunk_metadata = base_metadata.copy()
                chunk_metadata["chunk_index"] = chunk_index # 우리 시스템의 청크 순서 인덱스
                chunk_metadata["start_char"] = start
                chunk_metadata["end_char"] = end

                chunks.append(DocumentChunk(content=chunk_content, metadata=chunk_metadata))

                if end == text_len: break
                next_start = start + (self._fallback_chunk_size - self._fallback_chunk_overlap)
                start = max(start + 1, next_start)
                if start >= text_len: break
                chunk_index += 1


            print(f"DoclingChunkerAdapter: Chunking process finished. Generated {len(chunks)} chunks.")

            return chunks