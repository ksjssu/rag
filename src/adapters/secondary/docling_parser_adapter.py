# src/adapters/secondary/docling_parser_adapter.py

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import sys
import tempfile
import re
from dataclasses import dataclass, field
from enum import Enum

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
    class InputFormat(str, Enum):
        """A document format supported by document backend parsers."""
        DOCX = "docx"
        PPTX = "pptx"
        HTML = "html"
        IMAGE = "image"
        PDF = "pdf"
        ASCIIDOC = "asciidoc"
        MD = "md"
        CSV = "csv"
        XLSX = "xlsx"
        XML_USPTO = "xml_uspto"
        XML_JATS = "xml_jats"
        JSON_DOCLING = "json_docling"

        # __members__ 딕셔너리 업데이트 (UNKNOWN 제거)
        __members__ = {
            'DOCX': DOCX,
            'PPTX': PPTX,
            'HTML': HTML,
            'IMAGE': IMAGE,
            'PDF': PDF,
            'ASCIIDOC': ASCIIDOC,
            'MD': MD,
            'CSV': CSV,
            'XLSX': XLSX,
            'XML_USPTO': XML_USPTO,
            'XML_JATS': XML_JATS,
            'JSON_DOCLING': JSON_DOCLING
        }

        @classmethod
        def from_extension(cls, ext):
             ext_upper = ext.lstrip('.').upper()
             # Look up directly in the __members__ dictionary using the uppercase extension
             if ext_upper in cls.__members__: return cls.__members__[ext_upper]
             # Special mapping based on __members__ values
             # Access members via __members__ dictionary lookup
             if ext_upper in ['JPG', 'JPEG', 'PNG', 'TIFF'] and 'IMAGE' in cls.__members__: return cls.__members__['IMAGE']
             return cls.UNKNOWN if 'UNKNOWN' in cls.__members__ else None # Return dummy object value

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
DOCLING_ALLOWED_FORMATS = [
    "pdf", "docx", "xlsx", "pptx",
    "html", "md", "csv",
    "jpg", "jpeg", "png", "tif", "tiff", "bmp",
    "adoc", "xml", "json"
]

@dataclass
class ParsedDocument:
    """
    파싱 과정을 거쳐 텍스트 추출 및 기본 구조 정보가 포함된 문서 모델.
    """
    content: str  # 추출된 텍스트 내용
    metadata: Dict[str, Any] = field(default_factory=dict)  # 원본 메타데이터 + 파싱 중 얻은 메타데이터
    
    # 구조화된 콘텐츠
    tables: List[Dict[str, Any]] = field(default_factory=list)  # 테이블 구조 정보
    images: List[Dict[str, Any]] = field(default_factory=list)  # 이미지 정보
    equations: List[Dict[str, Any]] = field(default_factory=list)  # 수식 정보
    code_blocks: List[Dict[str, Any]] = field(default_factory=list)  # 코드 블록 정보
    
    # 문서 구조 정보
    headings: List[Dict[str, Any]] = field(default_factory=list)  # 제목 구조
    paragraphs: List[Dict[str, Any]] = field(default_factory=list)  # 단락 정보
    lists: List[Dict[str, Any]] = field(default_factory=list)  # 목록 정보
    
    # 레이아웃 정보
    layout: Optional[Dict[str, Any]] = None  # 페이지 레이아웃 정보
    page_info: List[Dict[str, Any]] = field(default_factory=list)  # 페이지별 정보
    
    # OCR 관련 정보
    ocr_results: Optional[Dict[str, Any]] = None  # OCR 결과 정보
    ocr_confidence: Optional[float] = None  # OCR 신뢰도
    
    # 이미지 처리 결과
    image_classifications: List[Dict[str, Any]] = field(default_factory=list)  # 이미지 분류 결과
    image_descriptions: List[Dict[str, Any]] = field(default_factory=list)  # 이미지 설명 결과
    
    # 테이블 처리 결과
    table_structures: List[Dict[str, Any]] = field(default_factory=list)  # 테이블 구조 분석 결과
    table_cell_matches: List[Dict[str, Any]] = field(default_factory=list)  # 테이블 셀 매칭 결과
    
    # 코드 및 수식 처리 결과
    code_enrichments: List[Dict[str, Any]] = field(default_factory=list)  # 코드 보강 정보
    formula_enrichments: List[Dict[str, Any]] = field(default_factory=list)  # 수식 보강 정보

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
                          # 이미지 확장자 처리
                          if fmt_upper in ['JPG', 'JPEG', 'PNG', 'TIF', 'TIFF', 'BMP']:
                              if InputFormat.IMAGE not in self._allowed_docling_formats:
                                  self._allowed_docling_formats.append(InputFormat.IMAGE)
                          # 다른 포맷 처리
                          elif fmt_upper in InputFormat.__members__:
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
    def _guess_input_format(self, filename: str) -> Optional[InputFormat]:
        """파일 확장자를 기반으로 InputFormat을 추측합니다."""
        ext = filename.lower().split('.')[-1]
        
        format_map = {
            'docx': InputFormat.DOCX,
            'dotx': InputFormat.DOCX,
            'docm': InputFormat.DOCX,
            'dotm': InputFormat.DOCX,
            'pptx': InputFormat.PPTX,
            'potx': InputFormat.PPTX,
            'ppsx': InputFormat.PPTX,
            'pptm': InputFormat.PPTX,
            'potm': InputFormat.PPTX,
            'ppsm': InputFormat.PPTX,
            'pdf': InputFormat.PDF,
            'md': InputFormat.MD,
            'html': InputFormat.HTML,
            'htm': InputFormat.HTML,
            'xhtml': InputFormat.HTML,
            'xml': InputFormat.XML_JATS,
            'nxml': InputFormat.XML_JATS,
            'jpg': InputFormat.IMAGE,
            'jpeg': InputFormat.IMAGE,
            'png': InputFormat.IMAGE,
            'tif': InputFormat.IMAGE,
            'tiff': InputFormat.IMAGE,
            'bmp': InputFormat.IMAGE,
            'adoc': InputFormat.ASCIIDOC,
            'asciidoc': InputFormat.ASCIIDOC,
            'asc': InputFormat.ASCIIDOC,
            'csv': InputFormat.CSV,
            'xlsx': InputFormat.XLSX,
            'json': InputFormat.JSON_DOCLING
        }
        
        input_format = format_map.get(ext)
        if input_format is None:
            raise ParsingError(f"지원하지 않는 파일 형식입니다: .{ext}")
        
        return input_format

    def parse(self, raw_document: RawDocument) -> ParsedDocument:
        """
        RawDocument를 Docling DocumentConverter로 파싱합니다.
        """
        filename = raw_document.metadata.get('filename', 'unknown')
        print(f"\n[PARSING] 시작: {filename}")

        if not raw_document.content:
            logger.warning("DoclingParserAdapter: Empty document content received")
            return ParsedDocument(content="", metadata=raw_document.metadata)

        if not _docling_available or not self._converter:
            logger.warning("DoclingParserAdapter: Using fallback parsing (Docling not available)")
            return ParsedDocument(content=raw_document.content, metadata=raw_document.metadata)

        try:
            # 입력 형식 확인 (수정된 부분)
            try:
                input_format = self._guess_input_format(filename)
                if not input_format:
                    raise ParsingError(f"파일 형식을 확인할 수 없습니다: {filename}")
            except ParsingError as e:
                logger.error(f"DoclingParserAdapter: {str(e)}")
                raise

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

            # 1. 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(raw_document.content)
                temp_path = Path(temp_file.name)

            # 2. DocumentConverter로 변환
            result = self._converter.convert(
                source=temp_path,
                headers={"Content-Type": "application/pdf"}
            )

            # 3. 텍스트 추출
            if result.status == ConversionStatus.SUCCESS:
                doc = result.document
                
                # 1. 기본 텍스트 추출
                backend_text = ""
                ocr_text = ""
                try:
                    if hasattr(doc, 'export_to_text'):
                        backend_text = doc.export_to_text()
                    if hasattr(doc, 'ocr_text'):
                        ocr_text = doc.ocr_text
                    elif hasattr(doc, 'export_to_markdown'):
                        ocr_text = doc.export_to_markdown()
                except Exception as e:
                    logger.error(f"Text extraction error: {e}")
                
                # 2. 테이블 정보 추출
                tables = []
                table_structures = []
                table_cell_matches = []
                try:
                    if hasattr(doc, 'tables'):
                        for table in doc.tables:
                            tables.append({
                                'structure': table.structure,
                                'content': table.content,
                                'position': table.position
                            })
                    if hasattr(doc, 'table_structures'):
                        table_structures = doc.table_structures
                    if hasattr(doc, 'table_cell_matches'):
                        table_cell_matches = doc.table_cell_matches
                except Exception as e:
                    logger.error(f"Table extraction error: {e}")
                
                # 3. 이미지 정보 추출
                images = []
                image_classifications = []
                image_descriptions = []
                try:
                    if hasattr(doc, 'images'):
                        for image in doc.images:
                            images.append({
                                'data': image.data,
                                'description': image.description,
                                'position': image.position
                            })
                    if hasattr(doc, 'image_classifications'):
                        image_classifications = doc.image_classifications
                    if hasattr(doc, 'image_descriptions'):
                        image_descriptions = doc.image_descriptions
                except Exception as e:
                    logger.error(f"Image extraction error: {e}")
                
                # 4. 수식 정보 추출
                equations = []
                formula_enrichments = []
                try:
                    if hasattr(doc, 'equations'):
                        for eq in doc.equations:
                            equations.append({
                                'content': eq.content,
                                'position': eq.position
                            })
                    if hasattr(doc, 'formula_enrichments'):
                        formula_enrichments = doc.formula_enrichments
                except Exception as e:
                    logger.error(f"Equation extraction error: {e}")
                
                # 5. 코드 블록 추출
                code_blocks = []
                code_enrichments = []
                try:
                    if hasattr(doc, 'code_blocks'):
                        for code in doc.code_blocks:
                            code_blocks.append({
                                'content': code.content,
                                'language': code.language,
                                'position': code.position
                            })
                    if hasattr(doc, 'code_enrichments'):
                        code_enrichments = doc.code_enrichments
                except Exception as e:
                    logger.error(f"Code block extraction error: {e}")
                
                # 6. 문서 구조 정보 추출
                headings = []
                paragraphs = []
                lists = []
                try:
                    if hasattr(doc, 'headings'):
                        headings = doc.headings
                    if hasattr(doc, 'paragraphs'):
                        paragraphs = doc.paragraphs
                    if hasattr(doc, 'lists'):
                        lists = doc.lists
                except Exception as e:
                    logger.error(f"Document structure extraction error: {e}")
                
                # 7. 레이아웃 정보 추출
                layout = None
                page_info = []
                try:
                    if hasattr(doc, 'layout'):
                        layout = doc.layout
                    if hasattr(doc, 'page_info'):
                        page_info = doc.page_info
                except Exception as e:
                    logger.error(f"Layout extraction error: {e}")
                
                # 8. OCR 정보 추출
                ocr_results = None
                ocr_confidence = None
                try:
                    if hasattr(doc, 'ocr_results'):
                        ocr_results = doc.ocr_results
                    if hasattr(doc, 'ocr_confidence'):
                        ocr_confidence = doc.ocr_confidence
                except Exception as e:
                    logger.error(f"OCR information extraction error: {e}")
                
                # 최종 텍스트 선택 및 정제
                document_text = ""
                if "glyph<" in backend_text or backend_text.strip() == "":
                    document_text = ocr_text
                    print("[PARSING] 글리프 감지됨: OCR 텍스트 사용")
                else:
                    document_text = backend_text
                    print("[PARSING] 정상 텍스트: 백엔드 텍스트 사용")
                
                # 텍스트 정제
                document_text = re.sub(r'glyph<[^>]+>', '', document_text)
                document_text = re.sub(r'<[^>]+>', '', document_text)
                
                print(f"[PARSING] 최종 추출 텍스트 길이: {len(document_text)} 글자")
                
                # 샘플 출력
                if document_text:
                    print("\n===== 추출된 텍스트 샘플 =====")
                    print(document_text[:200] + "..." if len(document_text) > 200 else document_text)
                    print("==============================\n")
                
                # 테이블 정보 추출
                tables = []
                if hasattr(doc, 'tables'):
                    for table in doc.tables:
                        tables.append({
                            'structure': table.structure,
                            'content': table.content,
                            'position': table.position
                        })
                
                # 이미지 정보 추출
                images = []
                if hasattr(doc, 'images'):
                    for image in doc.images:
                        images.append({
                            'data': image.data,
                            'description': image.description,
                            'position': image.position
                        })
                
                # 파싱 결과 출력 추가
                def print_parsing_results(parsed_doc: ParsedDocument):
                    print("\n========== 파싱 결과 ==========")
                    print(f"파일명: {filename}")
                    
                    # 기본 텍스트 내용
                    if parsed_doc.content:
                        print("\n[텍스트 내용 샘플]")
                        content_preview = parsed_doc.content[:200] + "..." if len(parsed_doc.content) > 200 else parsed_doc.content
                        print(content_preview)
                    
                    # 테이블 정보
                    if parsed_doc.tables:
                        print(f"\n[테이블 수]: {len(parsed_doc.tables)}")
                        for i, table in enumerate(parsed_doc.tables[:2], 1):  # 처음 2개만 출력
                            print(f"\n테이블 {i} 미리보기:")
                            print(f"- 구조: {table.get('structure', '정보 없음')}")
                            print(f"- 위치: {table.get('position', '정보 없음')}")
                    
                    # 이미지 정보
                    if parsed_doc.images:
                        print(f"\n[이미지 수]: {len(parsed_doc.images)}")
                        for i, img in enumerate(parsed_doc.images[:2], 1):  # 처음 2개만 출력
                            print(f"\n이미지 {i} 정보:")
                            print(f"- 설명: {img.get('description', '정보 없음')}")
                            print(f"- 위치: {img.get('position', '정보 없음')}")
                    
                    # 문서 구조
                    if parsed_doc.headings:
                        print(f"\n[제목 구조 수]: {len(parsed_doc.headings)}")
                        for i, heading in enumerate(parsed_doc.headings[:3], 1):  # 처음 3개만 출력
                            print(f"제목 {i}: {heading}")
                    
                    # OCR 결과
                    if parsed_doc.ocr_confidence is not None:
                        print(f"\n[OCR 신뢰도]: {parsed_doc.ocr_confidence:.2%}")
                    
                    # 수식 정보
                    if parsed_doc.equations:
                        print(f"\n[수식 수]: {len(parsed_doc.equations)}")
                        for i, eq in enumerate(parsed_doc.equations[:2], 1):  # 처음 2개만 출력
                            print(f"수식 {i}: {eq.get('content', '정보 없음')}")
                    
                    print("\n==============================\n")

                # FastAPI 응답 형식으로 변환
                def create_api_response(parsed_doc: ParsedDocument) -> dict:
                    return {
                        "filename": filename,
                        "content_preview": parsed_doc.content[:200] + "..." if len(parsed_doc.content) > 200 else parsed_doc.content,
                        "metadata": {
                            "tables_count": len(parsed_doc.tables),
                            "images_count": len(parsed_doc.images),
                            "headings_count": len(parsed_doc.headings),
                            "ocr_confidence": parsed_doc.ocr_confidence,
                            "equations_count": len(parsed_doc.equations)
                        },
                        "tables_preview": [
                            {
                                "structure": table.get('structure'),
                                "position": table.get('position')
                            } for table in parsed_doc.tables[:2]
                        ],
                        "images_preview": [
                            {
                                "description": img.get('description'),
                                "position": img.get('position')
                            } for img in parsed_doc.images[:2]
                        ],
                        "headings_preview": parsed_doc.headings[:3]
                    }

                # 결과 출력
                print_parsing_results(ParsedDocument(
                    content=document_text,
                    metadata=raw_document.metadata,
                    tables=tables,
                    images=images,
                    equations=equations,
                    code_blocks=code_blocks,
                    headings=headings,
                    paragraphs=paragraphs,
                    lists=lists,
                    layout=layout,
                    page_info=page_info,
                    ocr_results=ocr_results,
                    ocr_confidence=ocr_confidence,
                    image_classifications=image_classifications,
                    image_descriptions=image_descriptions,
                    table_structures=table_structures,
                    table_cell_matches=table_cell_matches,
                    code_enrichments=code_enrichments,
                    formula_enrichments=formula_enrichments
                ))
                
                # FastAPI 응답 데이터 저장
                raw_document.metadata['api_response'] = create_api_response(ParsedDocument(
                    content=document_text,
                    metadata=raw_document.metadata,
                    tables=tables,
                    images=images,
                    equations=equations,
                    code_blocks=code_blocks,
                    headings=headings,
                    paragraphs=paragraphs,
                    lists=lists,
                    layout=layout,
                    page_info=page_info,
                    ocr_results=ocr_results,
                    ocr_confidence=ocr_confidence,
                    image_classifications=image_classifications,
                    image_descriptions=image_descriptions,
                    table_structures=table_structures,
                    table_cell_matches=table_cell_matches,
                    code_enrichments=code_enrichments,
                    formula_enrichments=formula_enrichments
                ))
                
                return ParsedDocument(
                    content=document_text,
                    metadata=raw_document.metadata,
                    tables=tables,
                    images=images,
                    equations=equations,
                    code_blocks=code_blocks,
                    headings=headings,
                    paragraphs=paragraphs,
                    lists=lists,
                    layout=layout,
                    page_info=page_info,
                    ocr_results=ocr_results,
                    ocr_confidence=ocr_confidence,
                    image_classifications=image_classifications,
                    image_descriptions=image_descriptions,
                    table_structures=table_structures,
                    table_cell_matches=table_cell_matches,
                    code_enrichments=code_enrichments,
                    formula_enrichments=formula_enrichments
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

# PDF 파이프라인 옵션 설정
pdf_options = PdfPipelineOptions(
    # 1. OCR 관련 설정
    do_ocr=True,
    ocr_options=EasyOcrOptions(
        lang=["ko", "en"],
        confidence_threshold=0.3,
        download_enabled=True,
        force_full_page_ocr=True
    ),

    # 2. 테이블 처리 설정
    do_table_structure=True,
    table_structure_options=TableStructureOptions(
        do_cell_matching=True,
        mode=TableFormerMode.ACCURATE
    ),

    # 3. 레이아웃 분석 설정
    do_layout_analysis=True,
    layout_analysis_options={
        "detect_headers": True,
        "detect_footers": True,
        "detect_lists": True
    },

    # 4. 이미지 처리 설정
    do_picture_classification=True,
    do_picture_description=True,
    generate_page_images=True,
    generate_picture_images=True,
    images_scale=1.5,

    # 5. 텍스트 추출 설정
    force_backend_text=False,  # 내장 텍스트 우선 사용
    extract_text_from_figures=True,
    
    # 6. 수식/코드 인식 설정
    do_formula_enrichment=True,
    do_code_enrichment=True,

    # 7. 페이지 처리 설정
    generate_parsed_pages=True,
    
    # 8. 성능 설정
    accelerator_options=AcceleratorOptions(
        device="cpu",  # 또는 "cuda"
        num_threads=4
    )
)

# 파서 어댑터 생성 시 PDF 옵션 전달
parser_adapter = DoclingParserAdapter(
    allowed_formats=DOCLING_ALLOWED_FORMATS,
    pdf_options=pdf_options
)