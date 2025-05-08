# src/adapters/secondary/docling_parser_adapter.py

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import sys
import tempfile
import re

# Configure logging
logger = logging.getLogger(__name__)

# --- Docling 라이브러리 임포트 ---
# src/adapters/secondary/docling_parser_adapter.py

# --- Docling 라이브러리 임포트 ---
# 제공된 디렉토리 목록에 기반하여 정확한 임포트 경로로 수정합니다.
# 최상위 패키지는 'docling' 입니다.
try:
    # Docling 파싱을 위한 핵심 클래스 임포트
    # from docling_core.document_converter import DocumentConverter # 이전 시도 (실패)
    from docling.document_converter import DocumentConverter # <-- 정확한 경로: docling/document_converter.py

    # from docling_core.datamodel.base_models import DocumentStream, ConversionStatus, InputFormat # 이전 시도 (실패)
    from docling.datamodel.base_models import DocumentStream, ConversionStatus, InputFormat # <-- 정확한 경로: docling/datamodel/base_models.py

    # from docling_core.datamodel.document import ConversionResult # 이전 시도 (실패)
    from docling.datamodel.document import ConversionResult # <-- 정확한 경로: docling/datamodel/document.py

    # --- 파이프라인 옵션 임포트 ---
    # from docling_core.datamodel.pipeline_options import PipelineOptions, PdfPipelineOptions # 이전 시도 (실패)
    from docling.datamodel.pipeline_options import PipelineOptions, PdfPipelineOptions, EasyOcrOptions, TableStructureOptions, AcceleratorOptions, TableFormerMode # <-- 정확한 경로: docling/datamodel/pipeline_options.py

    # 필요한 다른 Docling 모듈/클래스 임포트 (예외 클래스 등)
    # from docling_core.exceptions import ConversionError as DoclingConversionError # 이전 시도 (실패)
    from docling.exceptions import ConversionError as DoclingConversionError # <-- 정확한 경로: docling/exceptions.py

    # Docling 자체 유틸리티 함수 임포트 (InputFormat 추정 등에 사용될 수 있음)
    # 예: from docling.utils.utils import guess_format_from_extension # <-- 정확한 경로: docling/utils/utils.py

    _docling_available = True
    print("Docling core libraries imported successfully.")
except ImportError as e: # 임포트 실패 시 발생하는 예외 메시지를 출력하도록 수정
    print(f"Warning: Docling library import failed. Import error: {e}") # <-- 실제 임포트 오류 메시지 출력
    print("DoclingParserAdapter will use fallback decoding.")
    _docling_available = False
    # --- Docling 클래스가 없을 경우 에러 방지를 위한 더미 클래스 정의 ---
    # (이전 더미 클래스 정의는 그대로 유지되어야 합니다.)
    print("   Using dummy Docling classes.")
    # Dummy 클래스 정의들...


# --- 어댑터 특정 예외 정의 ---
# ... (ParsingError 정의) ...

# ... (나머지 DoclingParserAdapter 클래스 코드 계속) ...

    # (Dummy DocumentConverter, Dummy DocumentStream 등 다른 더미 클래스 정의는 그대로 유지)

    # ★★★ 시작: 기존 더미 InputFormat 및 ConversionStatus 클래스 정의 전체와 이 블록 내용으로 교체 ★★★

    # Helper dummy class for InputFormat and ConversionStatus members (to simulate enum-like objects)
    class _DummyInputFormatMember:
        def __init__(self, name):
            self.name = name
        # Allow comparison with actual InputFormat/Status members or strings like 'PDF' or 'SUCCESS'
        def __eq__(self, other):
             if isinstance(other, _DummyInputFormatMember): return self.name == other.name
             # Allow comparison with uppercase strings
             if isinstance(other, str): return self.name == other.upper()
             # If comparing with actual Docling enum members, they might have a .name property
             if hasattr(other, 'name') and isinstance(getattr(other, 'name'), str): return self.name == getattr(other, 'name').upper()
             return False
        def __hash__(self): return hash(self.name) # Hash based on the name string
        def __str__(self): return self.name
        def __repr__(self): return f"<DummyStatusOrFormatMember:{self.name}>"

        # Needed for `if status in {SUCCESS, ...}` checks
        def __iter__(self): yield self # Allow iteration (e.g., when in a set)
        def __contains__(self, item): return item == self # Allow `in` operator check


    # Revised Dummy InputFormat (using helper class and simpler __members__)
    class InputFormat:
        # Define enum-like members using the helper class
        AUTODETECT = _DummyInputFormatMember('AUTODETECT')
        CSV = _DummyInputFormatMember('CSV')
        XLSX = _DummyInputFormatMember('XLSX')
        DOCX = _DummyInputFormatMember('DOCX')
        PPTX = _DummyInputFormatMember('PPTX')
        MD = _DummyInputFormatMember('MD')
        ASCIIDOC = _DummyInputFormatMember('ASCIIDOC')
        HTML = _DummyInputFormatMember('HTML')
        XML_USPTO = _DummyInputFormatMember('XML_USPTO')
        XML_JATS = _DummyInputFormatMember('XML_JATS')
        IMAGE = _DummyInputFormatMember('IMAGE')
        PDF = _DummyInputFormatMember('PDF')
        JSON_DOCLING = _DummyInputFormatMember('JSON_DOCLING')

        # Define __members__ dictionary mapping *string names* to the member objects
        # Use a standard dictionary comprehension after members are defined
        __members__ = {
            'AUTODETECT': AUTODETECT, 'CSV': CSV, 'XLSX': XLSX, 'DOCX': DOCX, 'PPTX': PPTX,
            'MD': MD, 'ASCIIDOC': ASCIIDOC, 'HTML': HTML, 'XML_USPTO': XML_USPTO, 'XML_JATS': XML_JATS,
            'IMAGE': IMAGE, 'PDF': PDF, 'JSON_DOCLING': JSON_DOCLING
        }

        @classmethod
        def from_extension(cls, ext):
             ext_upper = ext.lstrip('.').upper()
             # Look up directly in the __members__ dictionary using the uppercase extension
             if ext_upper in cls.__members__: return cls.__members__[ext_upper]
             # Special mapping based on __members__ values
             # Access members via __members__ dictionary lookup
             if ext_upper in ['JPG', 'JPEG', 'PNG', 'TIFF'] and 'IMAGE' in cls.__members__: return cls.__members__['IMAGE']
             return cls.AUTODETECT if 'AUTODETECT' in cls.__members__ else None # Return dummy object value

        # Add __getitem__ to allow dictionary-style access if needed elsewhere (e.g., InputFormat['PDF'])
        @classmethod
        def __getitem__(cls, name):
            name_upper = name.upper() # Allow case-insensitive access
            if name_upper in cls.__members__: return cls.__members__[name_upper]
            raise KeyError(f"'{name}' is not a valid dummy InputFormat member.")


    # Revised Dummy ConversionStatus enum (using helper class and simpler __members__)
    class ConversionStatus:
        SUCCESS = _DummyInputFormatMember('SUCCESS')
        PARTIAL_SUCCESS = _DummyInputFormatMember('PARTIAL_SUCCESS')
        FAILURE = _DummyInputFormatMember('FAILURE')
        SKIPPED = _DummyInputFormatMember('SKIPPED')
        # Define __members__ dictionary mapping *string names* to the member objects
        __members__ = {
            'SUCCESS': SUCCESS, 'PARTIAL_SUCCESS': PARTIAL_SUCCESS,
            'FAILURE': FAILURE, 'SKIPPED': SKIPPED
        }
        # Add __getitem__ for dictionary-style access if needed elsewhere
        @classmethod
        def __getitem__(cls, name):
            name_upper = name.upper()
            if name_upper in cls.__members__: return cls.__members__[name_upper]
            raise KeyError(f"'{name}' is not a valid dummy ConversionStatus member.")

        # Needed for `if status in {SUCCESS, ...}` checks (handled by _DummyInputFormatMember)
        # @property
        # def name(self): return self._name # Handled by _DummyInputFormatMember.name


    # Dummy ConversionResult
    class ConversionResult:
         def __init__(self, status, document=None, errors=None, warnings=None):
              self.status = status
              self.document = document
              self.errors = errors or []
              self.warnings = warnings or []
         @property
         def status(self): return self._status
         @status.setter
         def status(self, value): self._status = value
         @property
         def document(self): return self._document
         @document.setter
         def document(self, value): self._document = value


    # Dummy PipelineOptions and PdfPipelineOptions
    class PipelineOptions: pass
    class PdfPipelineOptions: pass

    # Dummy DoclingConversionError (needs to inherit from Exception)
    class DoclingConversionError(Exception): pass

    # Dummy guess_format_from_extension function if it's called directly (not used in current parse method)
    # def guess_format_from_extension(filename): return InputFormat.AUTODETECT # Simplified dummy


    # ★★★ 끝: 기존 더미 InputFormat 및 ConversionStatus 클래스 정의 전체와 이 블록 내용으로 교체 ★★★


# --- 어댑터 특정 예외 정의 ---
# 파싱 과정에서 발생하는 오류를 나타내기 위한 어댑터 레벨의 예외
class ParsingError(Exception):
    """Represents an error during the document parsing process."""
    pass

# ... (나머지 DoclingParserAdapter 클래스 코드 계속) ...


import os
from ports.output_ports import DocumentParsingPort # 구현할 포트 임포트
from domain.models import RawDocument, ParsedDocument # 입/출력 도메인 모델 임포트
from typing import Dict, Any, Optional, List, Union # Union 임포트
from pathlib import Path # 파일 확장자 추출에 사용

# src/adapters/secondary/docling_parser_adapter.py 파일 상단에 추가
DOCLING_ALLOWED_FORMATS = ["pdf", "docx", "xlsx", "pptx", "jpg", "png"]

class DoclingParserAdapter(DocumentParsingPort):
    """
    Docling 라이브러리(DocumentConverter)를 사용하여 DocumentParsingPort를 구현하는 어댑터.
    RawDocument를 Docling으로 파싱하고 결과를 ParsedDocument에 담아 반환합니다.
    """
    def __init__(
        self,
        allowed_formats: Optional[List[str]] = None, # 허용할 파일 형식 목록 (확장자 또는 Docling InputFormat 문자열)
        # Docling Converter 초기화에 필요한 다른 설정 파라미터가 있다면 여기에 추가
        # 예: api_key: str = None, config_path: str = None
        # 파이프라인 옵션 등 Docling 특정 설정을 어댑터 초기화 시 전달할 수 있습니다.
        # 예: pdf_options: Optional[PdfPipelineOptions] = None
        pdf_options: Optional[PdfPipelineOptions] = None # PDF 파이프라인 옵션 예시 (Docling 문서 확인)
        # 다른 형식에 대한 파이프라인 옵션도 유사하게 추가
        # image_options: Optional[ImagePipelineOptions] = None
    ):
        """
        DoclingParserAdapter 초기화. Docling DocumentConverter 인스턴스를 생성합니다.

        Args:
            allowed_formats: Docling에서 허용할 입력 파일 형식 목록 (확장자 또는 Docling InputFormat 문자열).
                             None이면 Docling 기본 설정 사용.
            pdf_options: PDF 파이프라인에 적용할 옵션 (Docling 문서 확인 필요).
            # 기타 Docling Converter 초기화 파라미터들 (Docling 문서 확인 필요)
        """
        self._converter: Optional[DocumentConverter] = None # Docling DocumentConverter 인스턴스
        self._allowed_docling_formats: Optional[List[InputFormat]] = None # Docling InputFormat enum 리스트
        self._pdf_options = pdf_options # PDF 파이프라인 옵션 저장 예시

        # Docling Converter 생성자에 전달할 format_options 딕셔너리 구성
        # PDFLoader 예시처럼 {InputFormat.PDF: PdfFormatOption(pipeline_options=...)} 형태
        # 다른 형식에 대한 옵션도 유사하게 추가 가능
        format_options_dict: Dict[InputFormat, FormatOption] = {} # FormatOption은 Docling 내부 클래스
        if _docling_available:
            # FormatOption 클래스를 Docling 라이브러리에서 임포트해야 합니다. (현재 임포트되지 않음)
            # from docling_core.document_converter import FormatOption, PdfFormatOption # <-- 임포트 필요
            # from docling_core.datamodel.pipeline_options import PdfPipelineOptions # <-- 임포트 필요
            # 더미 클래스를 사용하여 구조만 시뮬레이션
            class DummyFormatOption:
                def __init__(self, pipeline_cls=None, backend=None, pipeline_options=None):
                    self.pipeline_cls = pipeline_cls
                    self.backend = backend
                    self.pipeline_options = pipeline_options

            class DummyPdfFormatOption(DummyFormatOption):
                def __init__(self, pipeline_options=None, backend="pypdfium2"):
                    super().__init__(pipeline_cls=None, backend=backend, pipeline_options=pipeline_options)


            if self._pdf_options:
                 try:
                      # Docling InputFormat.PDF와 PdfFormatOption, PdfPipelineOptions 클래스가 필요합니다.
                      # Docling 문서에서 PdfFormatOption의 생성자 파라미터 확인 (pipeline_options를 받는지 등)
                      # format_options_dict[InputFormat.PDF] = PdfFormatOption(pipeline_options=self._pdf_options) # 실제 코드 형태
                      # 더미 클래스 사용 예시
                      if 'PDF' in InputFormat.__members__:
                          format_options_dict[InputFormat.PDF] = DummyPdfFormatOption(pipeline_options=self._pdf_options)
                          logger.info("Added PDF options to format_options_dict using dummy classes")
                 except Exception as e:
                      logger.warning(f"Could not configure PDF options for DoclingConverter: {e}")


        if _docling_available:
            self._is_initialized_successfully = False # 초기화 성공 여부 플래그
            logger.info("DoclingParserAdapter: Initializing Docling DocumentConverter...")
            try:
                 # --- Docling InputFormat 설정 (allowed_formats 처리) ---
                 # allowed_formats 문자열 목록을 Docling InputFormat enum으로 변환
                 if allowed_formats:
                      self._allowed_docling_formats = []
                      for fmt_str in allowed_formats:
                          fmt_upper = fmt_str.upper()
                          # Docling InputFormat.__members__에 있는지 확인하여 유효한 것만 추가
                          if fmt_upper in InputFormat.__members__:
                              self._allowed_docling_formats.append(InputFormat[fmt_upper])
                          else:
                              logger.warning(f"Specified allowed_format '{fmt_str}' is not a valid Docling InputFormat.")


                 # --- Docling DocumentConverter 인스턴스 생성 ★ 실제 초기화 ★ ---
                 # 제공된 Docling Converter 코드의 __init__ 시그니처에 맞춰 파라미터 전달
                 # allowed_formats와 format_options를 받는 것으로 보입니다.
                 # DoclingConverter 생성자가 받는 다른 파라미터 추가 (Docling 문서 확인)

                 self._converter = DocumentConverter(
                     allowed_formats=self._allowed_docling_formats,
                     format_options=format_options_dict, # 구성한 format_options 딕셔너리 전달
                     # 기타 Docling Converter 생성자가 받는 파라미터 추가 (Docling 문서 확인)
                 )
                 self._is_initialized_successfully = True # 초기화 성공
                 logger.info("DoclingParserAdapter: Docling DocumentConverter initialized successfully.")
            except Exception as e: # Docling Converter 초기화 중 발생할 수 있는 예외 처리
                logger.error(f"DoclingParserAdapter: Failed to initialize DocumentConverter: {e}")
                self._converter = None # 초기화 실패 시 None으로 설정
                # 초기화 실패 시 ParsingError 예외를 발생시켜 앱 시작 중단 고려
                # raise ParsingError(f"Failed to initialize Docling Converter: {e}") from e


        if self._converter is None:
             logger.warning("DoclingParserAdapter: Docling Converter not available or failed to initialize. Will use fallback decoding.")


    # --- Helper 메서드 ---
    def _guess_input_format(self, metadata: Dict[str, Any]) -> Optional[InputFormat]:
        """메타데이터에서 파일 확장자를 기반으로 Docling InputFormat을 추정합니다."""
        filename = metadata.get("filename", "")
        if not filename:
             return None # 파일명이 없으면 형식 추정 불가

        ext = Path(filename).suffix.lstrip('.').upper()

        # Docling InputFormat enum 멤버 이름을 순회하며 확장자와 일치하는지 확인
        # 제공된 코드의 _get_default_option 함수를 보면 확장자와 InputFormat 이름이 유사함을 알 수 있습니다.
        # 예: '.pdf' -> 'PDF', '.docx' -> 'DOCX'
        # 실제 Docling의 InputFormat.from_extension 또는 유사한 유틸리티 함수가 있다면 그것을 사용하는 것이 가장 정확하고 안전합니다.
        if _docling_available:
             try:
                 # Docling 자체 유틸리티 사용 시도 (from_extension 메서드가 있다고 가정)
                 if hasattr(InputFormat, 'from_extension') and callable(InputFormat.from_extension):
                     return InputFormat.from_extension(ext)
                 # Docling InputFormat.__members__ 직접 접근 시도 (덜 안전)
                 if ext in InputFormat.__members__:
                     return InputFormat[ext]
                 if ext in ['JPG', 'JPEG', 'PNG', 'TIFF'] and 'IMAGE' in InputFormat.__members__:
                     return InputFormat.IMAGE
                 # 다른 예외적인 매핑이 있다면 여기에 추가 (Docling 문서 확인)

             except Exception as e: # Docling 유틸리티 사용 중 오류 발생 시
                 logger.warning(f"Warning: Error guessing Docling InputFormat using utility: {e}. Falling back to manual guess.")
                 pass # 폴백 로직으로 이동


        # Docling 유틸리티가 없거나 실패 시 수동 매핑 시도 (정확하지 않을 수 있음)
        # InputFormat.__members__를 직접 사용하는 것은 Docling 버전에 따라 위험할 수 있습니다.
        # 가능한 Docling 자체의 포맷 추정 기능을 사용하거나, 명확한 매핑 테이블을 직접 만드는 것이 좋습니다.
        manual_mapping = {
            'PDF': InputFormat.PDF, 'DOCX': InputFormat.DOCX, 'XLSX': InputFormat.XLSX, 'PPTX': InputFormat.PPTX,
            'MD': InputFormat.MD, 'ASCIIDOC': InputFormat.ASCIIDOC, 'HTML': InputFormat.HTML,
            'CSV': InputFormat.CSV,
            'JPG': InputFormat.IMAGE, 'JPEG': InputFormat.IMAGE, 'PNG': InputFormat.IMAGE, 'TIFF': InputFormat.IMAGE,
            # 기타 필요한 매핑 추가
        }
        if ext in manual_mapping:
             return manual_mapping[ext]

        # 알 수 없을 경우 AUTODETECT 또는 None
        if 'AUTODETECT' in InputFormat.__members__:
             return InputFormat.AUTODETECT

        return None # 추정 실패 또는 지원하지 않는 형식

    def parse(self, raw_document: RawDocument) -> ParsedDocument:
        """
        RawDocument를 Docling DocumentConverter로 파싱합니다.
        """
        print(f"[PARSING] 시작: {raw_document.metadata.get('filename', 'unknown')}")

        if not raw_document.content:
            logger.warning("DoclingParserAdapter: Empty document content received")
            return ParsedDocument(content="", metadata=raw_document.metadata)

        if not _docling_available or not self._converter:
            logger.warning("DoclingParserAdapter: Using fallback parsing (Docling not available)")
            return ParsedDocument(content=raw_document.content, metadata=raw_document.metadata)

        try:
            input_format = self._guess_input_format(raw_document.metadata)
            if not input_format:
                logger.warning("DoclingParserAdapter: Could not determine input format, using AUTODETECT")
                input_format = InputFormat.AUTODETECT

            logger.info(f"DoclingParserAdapter: Converting document with format {input_format}")

            # 원본 파일명에서 확장자 추출 (metadata에 filename이 있다고 가정)
            filename = raw_document.metadata.get('filename', 'unknown_file')
            file_ext = os.path.splitext(filename)[1]  # '.pdf', '.docx' 등 확장자 추출

            # 확장자가 없는 경우 content-type에서 추론
            if not file_ext and 'content_type' in raw_document.metadata:
                content_type = raw_document.metadata['content_type']
                # MIME 타입별 확장자 매핑
                mime_to_ext = {
                    'application/pdf': '.pdf',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
                    # 기타 MIME 타입 추가...
                }
                file_ext = mime_to_ext.get(content_type, '')

            # 임시 파일 생성 (확장자 포함)
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(raw_document.content)
                temp_path = Path(temp_file.name)

            # 올바른 매개변수로 convert 호출
            result = self._converter.convert(
                source=temp_path,
                headers={"Content-Type": "application/pdf"}  # 필요한 경우
            )

            if result.status == ConversionStatus.SUCCESS:
                logger.info("DoclingParserAdapter: Document conversion successful")
                
                # 결과에서 문서 객체 가져오기
                doc = result.document
                
                # 1. 백엔드 텍스트 추출 시도
                backend_text = ""
                try:
                    if hasattr(doc, 'export_to_text') and callable(doc.export_to_text):
                        backend_text = doc.export_to_text()
                        print(f"[PARSING] 백엔드 텍스트 추출: {len(backend_text)} 글자")
                except Exception as e:
                    print(f"[PARSING] 백엔드 텍스트 추출 실패: {e}")
                
                # 2. OCR 텍스트 추출 시도
                ocr_text = ""
                try:
                    # 여러 방법으로 OCR 결과 접근 시도
                    if hasattr(doc, 'ocr_text') and doc.ocr_text:
                        ocr_text = doc.ocr_text
                    elif hasattr(doc, 'export_to_markdown') and callable(doc.export_to_markdown):
                        ocr_text = doc.export_to_markdown()
                    
                    if ocr_text:
                        print(f"[PARSING] OCR 텍스트 추출: {len(ocr_text)} 글자")
                except Exception as e:
                    print(f"[PARSING] OCR 텍스트 추출 실패: {e}")
                
                # 3. 최적의 텍스트 선택
                document_text = ""
                
                # 글리프 태그 확인
                if "glyph<" in backend_text or backend_text.strip() == "":
                    # 한글 또는 글리프 문제 - OCR 결과 사용
                    document_text = ocr_text
                    print("[PARSING] 글리프 감지됨: OCR 텍스트 사용")
                else:
                    # 일반 텍스트 PDF - 백엔드 텍스트 사용
                    document_text = backend_text
                    print("[PARSING] 정상 텍스트: 백엔드 텍스트 사용")
                
                # 텍스트 정제 (태그 제거 등)
                document_text = re.sub(r'glyph<[^>]+>', '', document_text)
                document_text = re.sub(r'<[^>]+>', '', document_text)
                
                print(f"[PARSING] 최종 추출 텍스트 길이: {len(document_text)} 글자")
                
                # 샘플 출력
                if document_text:
                    print("\n===== 추출된 텍스트 샘플 =====")
                    print(document_text[:200] + "..." if len(document_text) > 200 else document_text)
                    print("==============================\n")
                
                return ParsedDocument(
                    content=document_text,
                    metadata=raw_document.metadata
                )
            else:
                logger.error(f"DoclingParserAdapter: Document conversion failed with status {result.status}")
                if result.errors:
                    logger.error(f"Conversion errors: {result.errors}")
                raise ParsingError(f"Document conversion failed: {result.status}")

        except DoclingConversionError as e:
            logger.error(f"DoclingParserAdapter: Docling conversion error: {e}")
            raise ParsingError(f"Docling conversion failed: {e}") from e
        except Exception as e:
            logger.error(f"DoclingParserAdapter: Unexpected error during parsing: {e}")
            raise ParsingError(f"Unexpected error during parsing: {e}") from e

# PDF 파이프라인 옵션 설정 (DoclingParserAdapter 초기화 시)
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, TableStructureOptions, AcceleratorOptions, TableFormerMode

# OCR 옵션 설정 (한글 지원)
ocr_options = EasyOcrOptions(
    lang=["ko", "en"],
    confidence_threshold=0.3,
    download_enabled=True
)
# 생성 후 속성 설정
ocr_options.force_full_page_ocr = True

# 가속 옵션 설정
accelerator_options = AcceleratorOptions(
    device="cpu",  # 또는 "cuda" (GPU 사용 시)
    num_threads=4
)

# 테이블 구조 옵션
table_options = TableStructureOptions(
    do_cell_matching=True,
    mode=TableFormerMode.ACCURATE  # 정확한 테이블 추출 모드
)

# PDF 파이프라인 옵션 생성
pdf_options = PdfPipelineOptions(
    do_table_structure=True,      # 테이블 구조 추출
    do_ocr=True,                  # OCR 활성화
    do_code_enrichment=True,     # 코드 인식 활성화
    do_formula_enrichment=True,  # 수식 인식 활성화
    do_picture_classification=True,  # 이미지 분류 활성화
    do_picture_description=True,     # 이미지 설명 활성화
    force_backend_text=False,     # PDF 내장 텍스트 무시
    
    # 세부 옵션들
    table_structure_options=table_options,
    ocr_options=ocr_options,
    
    # 이미지 관련 설정
    images_scale=1.5,
    generate_page_images=True,    # OCR용 페이지 이미지 생성
    generate_picture_images=True,  # 문서 내 이미지 추출
    
    # 기타 설정
    generate_parsed_pages=True,  # 파싱된 페이지 정보 생성
    
    # 가속 옵션
    accelerator_options=accelerator_options
)

# 파서 어댑터 생성 시 옵션 전달
parser_adapter = DoclingParserAdapter(
    allowed_formats=DOCLING_ALLOWED_FORMATS,
    pdf_options=pdf_options
)