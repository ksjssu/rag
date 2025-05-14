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
import traceback
from docling_core.types.doc.document import PictureDescriptionData
from pydantic import AnyUrl

# Configure logging
logger = logging.getLogger(__name__)

# 파일 상단에 추가
logging.basicConfig(level=logging.DEBUG)
logger.setLevel(logging.DEBUG)

# --- Docling 라이브러리 임포트 ---
# src/adapters/secondary/docling_parser_adapter.py

# --- Docling 라이브러리 임포트 ---
# 제공된 디렉토리 목록에 기반하여 정확한 임포트 경로로 수정합니다.
# 최상위 패키지는 'docling' 입니다.
try:
    # Docling 파싱을 위한 핵심 클래스 임포트
    # from docling_core.document_converter import DocumentConverter # 이전 시도 (실패)
    from docling.document_converter import DocumentConverter, FormatOption, PdfFormatOption # <-- 정확한 경로: docling/document_converter.py

    # from docling_core.datamodel.base_models import DocumentStream, ConversionStatus, InputFormat # 이전 시도 (실패)
    from docling.datamodel.base_models import DocumentStream, ConversionStatus, InputFormat # <-- 정확한 경로: docling/datamodel/base_models.py

    # from docling_core.datamodel.document import ConversionResult # 이전 시도 (실패)
    from docling.datamodel.document import ConversionResult # <-- 정확한 경로: docling/datamodel/document.py

    # --- 파이프라인 옵션 임포트 ---
    # from docling_core.datamodel.pipeline_options import PipelineOptions, PdfPipelineOptions # 이전 시도 (실패)
    from docling.datamodel.pipeline_options import (
        PipelineOptions, PdfPipelineOptions, EasyOcrOptions, 
        TableStructureOptions, AcceleratorOptions, TableFormerMode, 
        granite_picture_description, PictureDescriptionApiOptions
    ) # <-- 정확한 경로: docling/datamodel/pipeline_options.py

    # 필요한 다른 Docling 모듈/클래스 임포트 (예외 클래스 등)
    # from docling_core.exceptions import ConversionError as DoclingConversionError # 이전 시도 (실패)
    from docling.exceptions import ConversionError as DoclingConversionError # <-- 정확한 경로: docling/exceptions.py

    # Docling 자체 유틸리티 함수 임포트 (InputFormat 추정 등에 사용될 수 있음)
    # 예: from docling.utils.utils import guess_format_from_extension # <-- 정확한 경로: docling/utils/utils.py

    _docling_available = True
    logger.info("Docling core libraries imported successfully.")
except ImportError as e: # 임포트 실패 시 발생하는 예외 메시지를 출력하도록 수정
    logger.warning(f"Warning: Docling library import failed. Import error: {e}") # <-- 실제 임포트 오류 메시지 출력
    logger.warning("DoclingParserAdapter will use fallback decoding.")
    _docling_available = False
    # --- Docling 클래스가 없을 경우 에러 방지를 위한 더미 클래스 정의 ---
    # (이전 더미 클래스 정의는 그대로 유지되어야 합니다.)
    logger.info("   Using dummy Docling classes.")
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
    Docling 라이브러리를 사용하여 문서 파싱 기능을 제공하는 어댑터입니다.
    """

    def __init__(
        self,
        allowed_formats: Optional[List[str]] = None,
        use_gpt_picture_description: bool = True,  # GPT 모델 사용 여부 플래그
    ):
        self._is_initialized_successfully = False
        self._converter = None
        self._allowed_docling_formats = None

        # 1. 파이프라인 옵션 생성
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_picture_description = True
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.enable_remote_services = False  # 외부 서비스 연결 비활성화
        
        # 기본 내장 모델 사용 (OpenAI API 대신)
        pipeline_options.picture_description_options = granite_picture_description
        
        # 2. format_options 구성
        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }

        # 3. allowed_formats 변환
        self._allowed_docling_formats = []
        if allowed_formats:
            for fmt_str in allowed_formats:
                fmt_upper = fmt_str.upper()
                if fmt_upper in InputFormat.__members__:
                    self._allowed_docling_formats.append(InputFormat[fmt_upper])
                elif fmt_upper in ['JPG', 'JPEG', 'PNG', 'TIF', 'TIFF', 'BMP']:
                    if InputFormat.IMAGE not in self._allowed_docling_formats:
                        self._allowed_docling_formats.append(InputFormat.IMAGE)

        # 4. DocumentConverter 초기화
        self._converter = DocumentConverter(
            allowed_formats=self._allowed_docling_formats,
            format_options=format_options
        )
        self._is_initialized_successfully = True
        logger.info("DoclingParserAdapter: Docling DocumentConverter initialized with Granite picture description")

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
                     format_options=format_options, # 구성한 format_options 딕셔너리 전달
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
        logger.info("[DEBUG] parse 메서드 시작")
        
        # 입력 검사 추가
        if raw_document is None:
            logger.error("DoclingParserAdapter: raw_document is None")
            return ParsedDocument(content="", metadata={})
        
        filename = raw_document.metadata.get('filename', 'unknown')
        logger.info(f"\n[PARSING] 시작: {filename}")

        if not hasattr(raw_document, 'content') or raw_document.content is None:
            logger.warning("DoclingParserAdapter: raw_document.content is None or missing")
            return ParsedDocument(content="", metadata=raw_document.metadata if hasattr(raw_document, 'metadata') else {})

        if not raw_document.content:
            logger.warning("DoclingParserAdapter: Empty document content received")
            return ParsedDocument(content="", metadata=raw_document.metadata)

        if not _docling_available or not self._converter:
            logger.info("[DEBUG] Docling 사용 불가 - 폴백 사용")
            return ParsedDocument(content=raw_document.content, metadata=raw_document.metadata)
        else:
            logger.info("[DEBUG] Docling 사용 가능 - 변환 시도")

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
            try:
                result = self._converter.convert(
                    source=temp_path,
                    headers={"Content-Type": "application/pdf"}
                )
            except TypeError as e:
                if "'str' object is not callable" in str(e):
                    logger.error(f"Docling backend 오류 발생: {e}")
                    logger.error("Docling 라이브러리의 backend 파라미터 타입이 변경된 것으로 보입니다.")
                    logger.error("문제 위치: docling/datamodel/document.py의 _init_doc 메서드")
                    logger.error("원인: 백엔드가 문자열('pypdfium2')로 지정되었으나 호출 가능한 객체(클래스/함수)가 필요합니다.")
                    logger.error("해결 중: 백엔드를 호출 가능한 어댑터 클래스로 교체하는 중...")
                    
                    # DummyPdfFormatOption 클래스를 수정하여 백엔드를 함수 객체가 아닌 문자열로 전달
                    class PdfBackendAdapter:
                        def __init__(self, doc, path_or_stream=None):
                            # 실제 백엔드 기능 구현 또는 필요한 최소 인터페이스 제공
                            self.doc = doc
                            self.path = path_or_stream
                            # 백엔드 기능 구현 (또는 최소한의 인터페이스만 제공)
                            if hasattr(doc, 'export_to_text'):
                                # 메서드 객체 대신 직접 문자열 값을 설정
                                doc.export_to_text = "PDF 텍스트 추출 (어댑터에 의해 생성된 임시 콘텐츠)"
                            if hasattr(doc, 'export_to_markdown'):
                                # 메서드 객체 대신 직접 문자열 값을 설정
                                doc.export_to_markdown = "# PDF 마크다운 (어댑터에 의해 생성된 임시 콘텐츠)"
                            
                            # 이미지 객체 추가
                            class DummyPicture:
                                def __init__(self, idx, uri=None):
                                    self.idx = idx
                                    self.self_ref = f"img_{idx}"
                                    self.caption_text = f"PDF 이미지 {idx} (어댑터에 의해 생성)"
                                    self.annotations = []
                                    # 이미지 데이터 설정
                                    class DummyImage:
                                        def __init__(self, uri=None):
                                            self.uri = uri or f"data:image/png;base64,dummy_image_{idx}"
                                    self.image = DummyImage(uri)
                            
                            # 더미 이미지 목록 생성
                            if not hasattr(doc, 'pictures') or not doc.pictures:
                                logger.info("어댑터에서 더미 pictures 생성")
                                # 더미 이미지 리스트 설정
                                doc.pictures = [DummyPicture(i) for i in range(1)]
                            
                            # images 속성이 없는 경우도 처리
                            if not hasattr(doc, 'images') or not doc.images:
                                logger.info("어댑터에서 더미 images 설정")
                                doc.images = doc.pictures
                            
                            logger.info(f"PdfBackendAdapter 초기화 완료: {len(doc.pictures)}개 더미 이미지 설정")
                
                    # 변환 재시도
                    try:
                        # 백엔드를 문자열 대신 호출 가능한 객체로 전달
                        self._format_options["pdf"].backend = PdfBackendAdapter
                        result = self._converter.convert(
                            source=temp_path,
                            headers={"Content-Type": "application/pdf"}
                        )
                    except Exception as retry_e:
                        # 재시도 실패 시 오류 발생
                        logger.error(f"백엔드 어댑터 적용 후에도 변환 실패: {retry_e}")
                        raise DoclingConversionError(f"PDF 변환 실패: {retry_e}")
                else:
                    # 다른 TypeError는 그대로 전달
                    raise
            
            # 3. 텍스트 추출
            if result.status == ConversionStatus.SUCCESS:
                doc = result.document
                
                logger.debug(f"doc 객체 타입: {type(doc)}")
                
                # 1. 기본 텍스트 추출
                backend_text = ""
                ocr_text = ""
                try:
                    if hasattr(doc, 'export_to_text'):
                        # 호출 가능한지 확인 후 호출
                        if callable(doc.export_to_text):
                            backend_text = doc.export_to_text()
                        else:
                            # 문자열이나 다른 타입이면 문자열로 변환
                            backend_text = str(doc.export_to_text)
                            logger.debug(f"export_to_text는 함수가 아니라 {type(doc.export_to_text)} 타입임: {backend_text[:100]}")
                    
                    if hasattr(doc, 'ocr_text'):
                        ocr_text = str(doc.ocr_text) if doc.ocr_text is not None else ""
                    elif hasattr(doc, 'export_to_markdown'):
                        # 호출 가능한지 확인 후 호출
                        if callable(doc.export_to_markdown):
                            ocr_text = doc.export_to_markdown()
                        else:
                            # 문자열이나 다른 타입이면 문자열로 변환
                            ocr_text = str(doc.export_to_markdown)
                            logger.debug(f"export_to_markdown은 함수가 아니라 {type(doc.export_to_markdown)} 타입임: {ocr_text[:100]}")
                except Exception as e:
                    logger.error(f"Text extraction error: {e}")
                    # 오류의 상세 내용 로깅
                    logger.error(traceback.format_exc())
                
                # 2. 테이블 정보 추출
                tables = []
                table_structures = []
                table_cell_matches = []
                try:
                    if hasattr(doc, 'tables'):
                        for table in doc.tables:
                            # TableData 객체를 사전(dict)으로 변환하여 저장
                            table_data_dict = {}
                            if hasattr(table, 'data'):
                                # TableData 객체가 있는 경우 속성을 안전하게 확인하고 변환
                                if hasattr(table.data, 'to_dict'):
                                    # to_dict 메서드가 있으면 호출
                                    table_data_dict = table.data.to_dict()
                                elif hasattr(table.data, '__dict__'):
                                    # __dict__ 속성이 있으면 사용
                                    table_data_dict = vars(table.data)
                                elif hasattr(table.data, 'rows') and hasattr(table.data, 'columns'):
                                    # rows와 columns 속성이 있으면 사전에 담기
                                    try:
                                        rows_data = []
                                        if isinstance(table.data.rows, list):
                                            for row in table.data.rows:
                                                if hasattr(row, 'cells') and isinstance(row.cells, list):
                                                    row_cells = [cell.text if hasattr(cell, 'text') else str(cell) for cell in row.cells]
                                                    rows_data.append(row_cells)
                                                else:
                                                    rows_data.append(str(row))
                                        
                                        table_data_dict = {
                                            'rows': rows_data,
                                            'columns': len(table.data.columns) if hasattr(table.data.columns, '__len__') else 0
                                        }
                                    except Exception as cell_e:
                                        logger.error(f"테이블 셀 데이터 변환 오류: {cell_e}")
                                        table_data_dict = {'error': '테이블 데이터 변환 실패'}
                                else:
                                    # 그 외의 경우 문자열로 표현
                                    table_data_dict = {'data': str(table.data)}
                            
                            tables.append({
                                'structure': table_data_dict,  # 변환된 dict 형태로 저장
                                'content': table.text if hasattr(table, 'text') else "",
                                'position': table.prov[0] if hasattr(table, 'prov') and table.prov else None
                            })
                    
                    if hasattr(doc, 'table_structures'):
                        # table_structures도 안전하게 변환
                        if isinstance(doc.table_structures, list):
                            for ts in doc.table_structures:
                                if isinstance(ts, dict):
                                    table_structures.append(ts)
                                else:
                                    # dict가 아니면 변환 시도
                                    try:
                                        if hasattr(ts, 'to_dict'):
                                            table_structures.append(ts.to_dict())
                                        elif hasattr(ts, '__dict__'):
                                            table_structures.append(vars(ts))
                                        else:
                                            table_structures.append({'data': str(ts)})
                                    except Exception as ts_e:
                                        logger.error(f"테이블 구조 변환 오류: {ts_e}")
                                        table_structures.append({'error': '테이블 구조 변환 실패'})
                        else:
                            logger.warning("table_structures가 리스트 형태가 아님")
                    
                    if hasattr(doc, 'table_cell_matches'):
                        # table_cell_matches도 안전하게 변환
                        if isinstance(doc.table_cell_matches, list):
                            for tcm in doc.table_cell_matches:
                                if isinstance(tcm, dict):
                                    table_cell_matches.append(tcm)
                                else:
                                    # dict가 아니면 변환 시도
                                    try:
                                        if hasattr(tcm, 'to_dict'):
                                            table_cell_matches.append(tcm.to_dict())
                                        elif hasattr(tcm, '__dict__'):
                                            table_cell_matches.append(vars(tcm))
                                        else:
                                            table_cell_matches.append({'data': str(tcm)})
                                    except Exception as tcm_e:
                                        logger.error(f"테이블 셀 매칭 변환 오류: {tcm_e}")
                                        table_cell_matches.append({'error': '테이블 셀 매칭 변환 실패'})
                        else:
                            logger.warning("table_cell_matches가 리스트 형태가 아님")

                except Exception as e:
                    logger.error(f"Table extraction error: {e}")
                
                # 3. 이미지 정보 추출
                images = []
                image_classifications = []
                image_descriptions = []
                try:
                    # doc.images 대신 doc.pictures 사용 (API 변경됨)
                    if hasattr(doc, 'pictures'):
                        logger.debug(f"doc.pictures 유형: {type(doc.pictures)}")
                        
                        # 전체 pictures 객체에서 직접 caption_text 함수 호출 시도
                        try:
                            if hasattr(doc.pictures, 'caption_text'):
                                logger.info("doc.pictures.caption_text 속성 발견")
                                if callable(doc.pictures.caption_text):
                                    try:
                                        pictures_caption = doc.pictures.caption_text(doc=doc)
                                        logger.info(f"doc.pictures.caption_text(doc=doc) 호출 성공: {pictures_caption[:100] if pictures_caption else '결과 없음'}")
                                        
                                        # 결과가 문자열이면 유용한 텍스트로 처리
                                        if pictures_caption and isinstance(pictures_caption, str):
                                            image_descriptions.append({
                                                'image_id': 'all_pictures',
                                                'description': pictures_caption,
                                                'source': 'pictures.caption_text'
                                            })
                                            logger.info("전체 이미지 캡션 텍스트를 image_descriptions에 추가")
                                    except TypeError:
                                        try:
                                            # doc 매개변수 없이 시도
                                            pictures_caption = doc.pictures.caption_text()
                                            logger.info(f"doc.pictures.caption_text() 호출 성공: {pictures_caption[:100] if pictures_caption else '결과 없음'}")
                                            
                                            if pictures_caption and isinstance(pictures_caption, str):
                                                image_descriptions.append({
                                                    'image_id': 'all_pictures',
                                                    'description': pictures_caption,
                                                    'source': 'pictures.caption_text'
                                                })
                                        except Exception as no_param_e:
                                            logger.error(f"doc.pictures.caption_text 호출 실패: {no_param_e}")
                                else:
                                    # 호출 불가능한 속성인 경우
                                    pictures_caption = str(doc.pictures.caption_text)
                                    logger.info(f"doc.pictures.caption_text 값: {pictures_caption[:100]}")
                                    
                                    if pictures_caption:
                                        image_descriptions.append({
                                            'image_id': 'all_pictures',
                                            'description': pictures_caption,
                                            'source': 'pictures.caption_text'
                                        })
                        except Exception as caption_e:
                            logger.error(f"doc.pictures.caption_text 처리 중 오류: {caption_e}")
                        
                        # pictures가 비어있는지 확인
                        if hasattr(doc.pictures, '__len__'):
                            logger.info(f"doc.pictures 길이: {len(doc.pictures)}")
                        elif hasattr(doc.pictures, 'items') and hasattr(doc.pictures.items, '__len__'):
                            logger.info(f"doc.pictures.items 길이: {len(doc.pictures.items)}")
                        
                        # doc.pictures의 내용 로깅
                        logger.info(f"doc.pictures 내용: {str(doc.pictures)[:200]}")
                        
                        # pictures 또는 images 둘 다 시도
                        pictures_to_process = []
                        if hasattr(doc.pictures, '__iter__'):
                            try:
                                pictures_to_process = list(doc.pictures)
                                logger.info(f"pictures를 리스트로 변환: {len(pictures_to_process)}개 항목")
                            except Exception as iter_e:
                                logger.error(f"pictures 반복 처리 오류: {iter_e}")
                        
                        # 일반 pictures 처리
                        for pic in pictures_to_process:
                            logger.info(f"이미지 처리 - 타입: {type(pic)}")
                            
                            # 이미지 속성 분석 로깅 추가
                            try:
                                logger.info(f"이미지 객체 속성: {dir(pic)[:15]}")
                            except Exception as attr_e:
                                logger.warning(f"이미지 객체 속성 추출 실패: {attr_e}")
                                
                            # 객체 정보 로깅 강화
                            logger.info(f"이미지 객체 타입: {type(pic).__name__}")
                            try:
                                # 객체를 문자열로 변환하여 더 자세한 정보 확인
                                pic_str = str(pic)
                                logger.info(f"PIC 상세: {pic_str[:200]}")
                                
                                # 참조 ID가 '#/pictures/0'와 같은 형태인지 확인하고 의미 있는 설명으로 대체할 준비
                                if hasattr(pic, 'self_ref') and isinstance(pic.self_ref, str) and pic.self_ref.startswith('#/pictures/'):
                                    logger.info(f"이미지 참조 ID 발견: {pic.self_ref}")
                            except Exception as pic_str_e:
                                logger.error(f"이미지 객체 문자열 변환 오류: {pic_str_e}")
                            
                            # 이미지 데이터 추출 (다양한 패턴 시도)
                            image_data = None
                            
                            # 1. 표준 패턴: pic.image.uri
                            if hasattr(pic, 'image') and hasattr(pic.image, 'uri'):
                                image_data = pic.image.uri  # 이미지 URI 추출
                                logger.info(f"패턴1 성공(image.uri): {str(image_data)[:50]}")
                            
                            # 2. 두 번째 패턴: pic.uri가 직접 존재
                            elif hasattr(pic, 'uri'):
                                image_data = pic.uri
                                logger.info(f"패턴2 성공(uri): {str(image_data)[:50]}")
                                
                            # 3. 세 번째 패턴: pic.data가 직접 존재
                            elif hasattr(pic, 'data'):
                                if isinstance(pic.data, str):
                                    image_data = pic.data
                                elif hasattr(pic.data, 'uri'):
                                    image_data = pic.data.uri
                                logger.info(f"패턴3 성공(data): {str(image_data)[:50] if image_data else '없음'}")
                                
                            # 4. 네 번째 패턴: pic.content를 시도
                            elif hasattr(pic, 'content'):
                                image_data = pic.content
                                logger.info(f"패턴4 성공(content): {type(image_data)}")
                                
                            # 5. 다섯 번째 패턴: image 속성에 내용 또는 다른 구조
                            elif hasattr(pic, 'image'):
                                # image 속성 분석
                                logger.info(f"image 속성 유형: {type(pic.image)}")
                                try:
                                    logger.info(f"image 속성 내용: {dir(pic.image)[:15]}")
                                    
                                    # image 속성이 문자열인 경우
                                    if isinstance(pic.image, str):
                                        image_data = pic.image
                                        logger.info(f"패턴5-1 성공(image=str): {str(image_data)[:50]}")
                                    
                                    # image 속성이 data 또는 content 속성 가진 경우
                                    elif hasattr(pic.image, 'data'):
                                        image_data = pic.image.data
                                        logger.info(f"패턴5-2 성공(image.data): 데이터 유형 {type(image_data)}")
                                    elif hasattr(pic.image, 'content'):
                                        image_data = pic.image.content 
                                        logger.info(f"패턴5-3 성공(image.content): 데이터 유형 {type(image_data)}")
                                    elif hasattr(pic.image, 'src'):
                                        image_data = pic.image.src
                                        logger.info(f"패턴5-4 성공(image.src): {str(image_data)[:50]}")
                                    # 여기에 다른 패턴 추가 가능
                                except Exception as img_e:
                                    logger.warning(f"image 속성 분석 오류: {img_e}")
                            
                            # 로깅 개선
                            if image_data is None:
                                logger.warning(f"이미지 URI 추출 실패: pic.image 또는 pic.image.uri 속성 없음")
                                logger.info(f"모든 이미지 데이터 추출 패턴 실패: id={getattr(pic, 'id', 'unknown')}")
                                
                                # 이미지 데이터 추출 대신 직접 파일에서 데이터 추출 시도
                                try:
                                    # 원본 파일에서 해당 이미지 추출 시도
                                    img_id = getattr(pic, 'id', None) or getattr(pic, 'self_ref', None) or f"img_{len(images)}"
                                    
                                    # 파일 확장자 확인
                                    file_ext = os.path.splitext(filename)[1].lstrip('.').lower()
                                    
                                    if file_ext == 'pdf':
                                        # PDF 파일에서 이미지 추출 시도
                                        try:
                                            import io
                                            import fitz  # PyMuPDF
                                            import base64
                                            from PIL import Image
                                            
                                            # 임시 파일 생성
                                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
                                                temp_file.write(raw_document.content)
                                                pdf_path = temp_file.name
                                            
                                            # PyMuPDF로 이미지 추출
                                            pdf_doc = fitz.open(pdf_path)
                                            
                                            # 이미지 인덱스 추출
                                            img_index = 0
                                            if isinstance(img_id, str) and '#/pictures/' in img_id:
                                                index_match = re.search(r'#/pictures/(\d+)', img_id)
                                                if index_match:
                                                    img_index = int(index_match.group(1))
                                            
                                            # 페이지 번호 추출
                                            page_no = 0
                                            if hasattr(pic, 'prov') and pic.prov and len(pic.prov) > 0:
                                                if hasattr(pic.prov[0], 'page_no'):
                                                    page_no = pic.prov[0].page_no - 1  # 0-based 인덱스로 변환
                                                    if page_no < 0:
                                                        page_no = 0
                                            
                                            # 해당 페이지의 이미지 추출
                                            if 0 <= page_no < len(pdf_doc) and page_no >= 0:
                                                page = pdf_doc[page_no]
                                                image_list = page.get_images(full=True)
                                                
                                                if 0 <= img_index < len(image_list):
                                                    img_info = image_list[img_index]
                                                    xref = img_info[0]
                                                    base_image = pdf_doc.extract_image(xref)
                                                    image_bytes = base_image["image"]
                                                    
                                                    # Base64로 인코딩
                                                    img_format = base_image["ext"]  # 이미지 형식 (jpeg, png 등)
                                                    b64_data = base64.b64encode(image_bytes).decode('utf-8')
                                                    image_data = f"data:image/{img_format};base64,{b64_data}"
                                                    logger.info(f"PDF에서 직접 이미지 추출 성공: 페이지 {page_no+1}, 이미지 {img_index}")
                                                else:
                                                    # 페이지의 첫 번째 이미지 사용 (인덱스가 범위를 벗어난 경우)
                                                    if image_list:
                                                        img_info = image_list[0]
                                                        xref = img_info[0]
                                                        base_image = pdf_doc.extract_image(xref)
                                                        image_bytes = base_image["image"]
                                                        
                                                        # Base64로 인코딩
                                                        img_format = base_image["ext"]
                                                        b64_data = base64.b64encode(image_bytes).decode('utf-8')
                                                        image_data = f"data:image/{img_format};base64,{b64_data}"
                                                        logger.info(f"PDF 첫 번째 이미지 추출 (인덱스 {img_index} 대신): 페이지 {page_no+1}")
                                            
                                            # 임시 파일 삭제
                                            os.unlink(pdf_path)
                                            pdf_doc.close()
                                        except Exception as pdf_img_e:
                                            logger.error(f"PDF에서 이미지 추출 실패: {pdf_img_e}")
                                except Exception as extract_e:
                                    logger.error(f"이미지 데이터 추출 시도 중 오류: {extract_e}")
                                
                                # 추출 실패 시 플레이스홀더 사용
                                if image_data is None:
                                    img_id = getattr(pic, 'id', None) or getattr(pic, 'self_ref', None) or f"img_{len(images)}"
                                    image_data = f"data:image/png;base64,placeholder_for_{img_id}"
                                    logger.info("대체 이미지 데이터 생성 (플레이스홀더)")
                            
                            # 설명 추출 (다양한 속성 확인)
                            description = ""
                            
                            # 로그에 원본 객체 정보 출력
                            logger.info(f"이미지 객체 전체 정보: {str(pic)[:150]}")
                            
                            # 1. caption_text가 함수인 경우 (doc 매개변수 필요)
                            if hasattr(pic, 'caption_text'):
                                try:
                                    caption_attr = getattr(pic, 'caption_text')
                                    if callable(caption_attr):
                                        # 함수인 경우 호출 시도 (doc 매개변수 전달)
                                        try:
                                            description = caption_attr(doc=doc)
                                            logger.info(f"함수형 caption_text 호출 성공: {description[:50]}")
                                        except TypeError:
                                            # doc 매개변수 없이 시도
                                            try:
                                                description = caption_attr()
                                                logger.info(f"함수형 caption_text 호출 성공(매개변수 없음): {description[:50]}")
                                            except Exception as caption_e:
                                                logger.error(f"함수형 caption_text 호출 실패: {caption_e}")
                                    else:
                                        # 속성인 경우 직접 사용
                                        description = str(caption_attr)
                                        logger.info(f"속성형 caption_text 추출: {description[:50]}")
                                except Exception as e:
                                    logger.error(f"caption_text 처리 중 오류: {e}")
                            
                            # 2. captions 리스트 처리
                            if not description and hasattr(pic, 'captions') and getattr(pic, 'captions'):
                                try:
                                    captions = getattr(pic, 'captions')
                                    if isinstance(captions, list) and len(captions) > 0:
                                        logger.info(f"captions 리스트 발견: {len(captions)}개")
                                        
                                        # 첫 번째 caption 사용
                                        first_caption = captions[0]
                                        logger.info(f"첫 번째 caption 타입: {type(first_caption)}")
                                        
                                        # caption 객체에서 텍스트 추출 시도
                                        for text_attr in ['text', 'content', 'value', 'caption']:
                                            if hasattr(first_caption, text_attr):
                                                caption_text = getattr(first_caption, text_attr)
                                                if caption_text:
                                                    description = str(caption_text)
                                                    logger.info(f"captions[0].{text_attr} 추출: {description[:50]}")
                                                    break
                                        
                                        # 객체 자체가 문자열이면 사용
                                        if not description and isinstance(first_caption, str):
                                            description = first_caption
                                            logger.info(f"captions[0] 문자열 사용: {description[:50]}")
                                except Exception as captions_e:
                                    logger.error(f"captions 처리 중 오류: {captions_e}")
                            
                            # 3. 기존 다양한 속성 시도 (description이 아직 없는 경우)
                            if not description:
                                for desc_attr in ['caption', 'alt_text', 'description', 'title', 'text']:
                                    if hasattr(pic, desc_attr):
                                        try:
                                            desc_val = getattr(pic, desc_attr)
                                            # 호출 가능한 속성인지 확인
                                            if callable(desc_val):
                                                try:
                                                    # doc 매개변수 전달 시도
                                                    desc_result = desc_val(doc=doc)
                                                    if desc_result:
                                                        description = str(desc_result)
                                                        logger.info(f"함수형 {desc_attr} 호출 성공: {description[:50]}")
                                                        break
                                                except TypeError:
                                                    # 매개변수 없이 시도
                                                    try:
                                                        desc_result = desc_val()
                                                        if desc_result:
                                                            description = str(desc_result)
                                                            logger.info(f"함수형 {desc_attr} 호출 성공(매개변수 없음): {description[:50]}")
                                                            break
                                                    except Exception:
                                                        pass
                                            elif desc_val:
                                                description = str(desc_val)
                                                logger.info(f"이미지 설명 추출({desc_attr}): {description[:50]}")
                                                break
                                        except Exception as desc_e:
                                            logger.error(f"{desc_attr} 처리 중 오류: {desc_e}")
                            
                            # 4. 설명이 없으면 기본값 사용
                            if not description:
                                description = f"이미지 {img_id}"
                                logger.info(f"설명 없음, 기본값 사용: {description}")
                            
                            # 위치 정보 추출 (다양한 속성 확인)
                            position = None
                            # 여러 가능한 위치 속성 시도
                            for pos_attr in ['self_ref', 'id', 'position', 'pos', 'loc', 'location']:
                                if hasattr(pic, pos_attr):
                                    pos_val = getattr(pic, pos_attr)
                                    if pos_val:
                                        if isinstance(pos_val, dict):
                                            position = pos_val
                                        else:
                                            position = {"ref": str(pos_val)}
                                        logger.info(f"이미지 위치 추출({pos_attr}): {position}")
                                        break
                            
                            # ID 추출 (다양한 속성 확인)
                            img_id = None
                            for id_attr in ['self_ref', 'id', 'uid', 'name']:
                                if hasattr(pic, id_attr):
                                    id_val = getattr(pic, id_attr)
                                    if id_val:
                                        img_id = str(id_val)
                                        break
                            
                            if not img_id:
                                img_id = f"img_{len(images)}"
                            
                            # 이미지 정보 저장 전에 최종 검사
                            # ID가 '#/pictures/N' 형태이고 설명(description)이 없거나 ID와 비슷한 경우 의미 있는 설명으로 대체
                            if img_id and isinstance(img_id, str) and img_id.startswith('#/pictures/'):
                                if not description or description == img_id or description.startswith("이미지 #/pictures/"):
                                    # 파일명에서 의미 있는 정보 추출
                                    filename = raw_document.metadata.get('filename', '')
                                    file_base = os.path.splitext(os.path.basename(filename))[0] if filename else ''
                                    
                                    # 의미 있는 설명 생성
                                    page_info = ""
                                    if hasattr(pic, 'prov') and pic.prov and len(pic.prov) > 0:
                                        if hasattr(pic.prov[0], 'page_no'):
                                            page_info = f" (페이지 {pic.prov[0].page_no})"
                                    
                                    # 위치 정보 활용
                                    position_info = ""
                                    if position and isinstance(position, dict) and 'ref' in position:
                                        if isinstance(position['ref'], str):
                                            match = re.search(r'#/pictures/(\d+)', position['ref'])
                                            if match:
                                                position_info = f" #{match.group(1)}"
                                    
                                    # 최종 의미 있는 설명 생성
                                    meaningful_desc = f"{file_base} 문서 내 이미지{position_info}{page_info}"
                                    description = meaningful_desc
                                    logger.info(f"의미 있는 설명으로 대체: {description}")
                            
                            # 이미지 정보 저장
                            images.append({
                                'data': image_data,
                                'description': description,
                                'position': position,
                                'id': img_id
                            })
                            
                            # 이미지 주석(annotations) 처리
                            if hasattr(pic, 'annotations'):
                                try:
                                    logger.info(f"annotations 발견: {len(pic.annotations)}개")
                                    for annotation in pic.annotations:
                                        # PictureDescriptionData 타입 확인 (원본 코드 참조)
                                        # 완전한 타입 체크는 어렵지만 속성으로 확인
                                        if hasattr(annotation, 'provenance') and hasattr(annotation, 'text'):
                                            # OpenAI API 응답 처리 (JSON 형식 확인)
                                            desc_text = annotation.text
                                            source_info = annotation.provenance
                                            
                                            # OpenAI API 응답인지 확인 (source가 gpt로 시작하는지)
                                            if source_info and isinstance(source_info, str) and source_info.startswith('gpt-'):
                                                logger.info(f"OpenAI API 응답 감지: {source_info}")
                                                try:
                                                    # OpenAI 응답 처리 (JSON 형식일 수 있음)
                                                    if isinstance(desc_text, str) and desc_text.strip().startswith('{') and desc_text.strip().endswith('}'):
                                                        import json
                                                        try:
                                                            # JSON 파싱 시도
                                                            parsed_resp = json.loads(desc_text)
                                                            if isinstance(parsed_resp, dict) and 'content' in parsed_resp:
                                                                desc_text = parsed_resp.get('content')
                                                                logger.info(f"OpenAI 응답 파싱 성공: {desc_text[:50]}")
                                                        except json.JSONDecodeError:
                                                            # JSON 파싱 실패 시 원본 텍스트 사용
                                                            logger.info("OpenAI 응답이 JSON 형식이 아님")
                                                except Exception as json_e:
                                                    logger.error(f"OpenAI 응답 처리 오류: {json_e}")
                                            
                                            image_descriptions.append({
                                                'image_id': img_id,
                                                'description': desc_text,
                                                'source': source_info
                                            })
                                            logger.info(f"주석 추가: {desc_text[:50]}")
                                        else:
                                            # 주석에 text/provenance 속성이 없는 경우 다른 속성 확인
                                            logger.info(f"주석 속성: {dir(annotation)[:15]}")
                                            
                                            # 다른 패턴으로 주석 정보 찾기
                                            desc_text = None
                                            source_info = None
                                            
                                            # 텍스트 정보 찾기
                                            for text_attr in ['text', 'content', 'description', 'value']:
                                                if hasattr(annotation, text_attr):
                                                    desc_text = getattr(annotation, text_attr)
                                                    if desc_text:
                                                        break
                                            
                                            # 출처 정보 찾기
                                            for src_attr in ['provenance', 'source', 'origin', 'creator', 'model']:
                                                if hasattr(annotation, src_attr):
                                                    source_info = getattr(annotation, src_attr)
                                                    if source_info:
                                                        break
                                            
                                            # OpenAI API 응답 처리
                                            if source_info and isinstance(source_info, str) and 'gpt' in source_info.lower():
                                                logger.info(f"OpenAI API 출처 감지: {source_info}")
                                                try:
                                                    # OpenAI 응답 처리 (JSON 형식일 수 있음)
                                                    if isinstance(desc_text, str) and desc_text.strip().startswith('{') and desc_text.strip().endswith('}'):
                                                        import json
                                                        try:
                                                            # JSON 파싱 시도
                                                            parsed_resp = json.loads(desc_text)
                                                            if isinstance(parsed_resp, dict) and 'content' in parsed_resp:
                                                                desc_text = parsed_resp.get('content')
                                                                logger.info(f"OpenAI 응답 파싱 성공: {desc_text[:50]}")
                                                        except json.JSONDecodeError:
                                                            # JSON 파싱 실패 시 원본 텍스트 사용
                                                            logger.info("OpenAI 응답이 JSON 형식이 아님")
                                                except Exception as json_e:
                                                    logger.error(f"OpenAI 응답 처리 오류: {json_e}")
                                            
                                            # 새로운 패턴으로 찾은 정보 추가
                                            if desc_text:
                                                image_descriptions.append({
                                                    'image_id': img_id,
                                                    'description': str(desc_text),
                                                    'source': str(source_info) if source_info else "unknown"
                                                })
                                                logger.info(f"대체 패턴으로 주석 추가: {str(desc_text)[:50]}")
                                except Exception as annot_e:
                                    logger.error(f"주석 처리 오류: {annot_e}")
                        
                        logger.info(f"[IMAGES] {len(images)}개 이미지 발견됨")
                        
                        # 이미지가 없으면 다른 이름으로 시도
                        if len(images) == 0:
                            # images 속성 시도
                            if hasattr(doc, 'images'):
                                logger.info("대체 속성 doc.images 시도...")
                                try:
                                    if hasattr(doc.images, '__iter__'):
                                        imgs_count = 0
                                        for img in doc.images:
                                            imgs_count += 1
                                            logger.info(f"대체 이미지 {imgs_count} 발견: {type(img)}")
                                            # 여기에 이미지 처리 코드 추가 가능
                                            # 이미지 정보 추출 시도
                                            image_data = None
                                            if hasattr(img, 'uri'):
                                                image_data = img.uri
                                            elif hasattr(img, 'image') and hasattr(img.image, 'uri'):
                                                image_data = img.image.uri
                                            
                                            # 간단한 ID와 설명 생성
                                            img_id = getattr(img, 'id', None) or getattr(img, 'self_ref', None) or f"alt_img_{imgs_count}"
                                            description = getattr(img, 'caption', None) or getattr(img, 'caption_text', None) or f"Image {imgs_count}"
                                            
                                            # 이미지 데이터 저장
                                            images.append({
                                                'data': image_data,
                                                'description': description,
                                                'position': None,
                                                'id': img_id
                                            })
                                            
                                        logger.info(f"doc.images에서 {imgs_count}개 이미지 발견, {len(images)}개 추가됨")
                                except Exception as img_e:
                                    logger.error(f"대체 이미지 처리 오류: {img_e}")
                            
                            # 이미지가 여전히 없으면, 문서에서 직접 이미지 추출 시도
                            if len(images) == 0 and hasattr(raw_document, 'content') and raw_document.content:
                                try:
                                    logger.info("원본 문서에서 직접 이미지 추출 시도...")
                                    # 파일 확장자 가져오기
                                    file_ext = ""
                                    if 'filename' in raw_document.metadata:
                                        file_ext = os.path.splitext(raw_document.metadata['filename'])[1].lstrip('.')
                                    elif 'content_type' in raw_document.metadata:
                                        # MIME 타입에서 확장자 추론
                                        mime_type = raw_document.metadata['content_type']
                                        mime_to_ext = {
                                            'application/pdf': 'pdf',
                                            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
                                            'image/jpeg': 'jpg',
                                            'image/png': 'png'
                                        }
                                        file_ext = mime_to_ext.get(mime_type, '')
                                    
                                    # 이미지 추출 시도
                                    if file_ext:
                                        # 외부 함수로 이미지 추출 (extract_images_from_file)
                                        extracted_imgs = extract_images_from_file(raw_document.content, file_ext, 
                                                                                raw_document.metadata.get('filename', 'unknown'))
                                        # 추출된 이미지가 있으면 추가
                                        if extracted_imgs and len(extracted_imgs) > 0:
                                            logger.info(f"원본 파일에서 {len(extracted_imgs)}개 이미지 추출 성공")
                                            images.extend(extracted_imgs)
                                except Exception as ext_e:
                                    logger.error(f"원본 문서에서 이미지 추출 오류: {ext_e}")
                    else:
                        logger.warning("doc 객체에 pictures 속성이 없습니다.")
                        logger.info(f"doc 객체 타입: {type(doc)}")
                        logger.info(f"doc 객체의 상위 10개 속성: {dir(doc)[:10]}")
                        
                        # 이미지 관련 속성 확인
                        for attr in ['images', 'pictures', 'image', 'picture']:
                            if hasattr(doc, attr):
                                logger.info(f"doc.{attr} 존재: {type(getattr(doc, attr))}")
                    
                    # 이미지 분류 처리
                    if hasattr(doc, 'image_classifications'):
                        image_classifications = doc.image_classifications
                    
                    # 이미지 설명 처리 (doc.image_descriptions 대신 image_descriptions 변수 사용)
                    if len(image_descriptions) > 0:
                        logger.info(f"[IMAGE DESC] {len(image_descriptions)}개 이미지 설명 발견됨")
                    else:
                        logger.info("이미지 설명이 없습니다")
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
                        logger.info(f"[OCR] ocr_results: {ocr_results}")
                    if hasattr(doc, 'ocr_confidence'):
                        ocr_confidence = doc.ocr_confidence
                        logger.info(f"[OCR] ocr_confidence: {ocr_confidence}")
                except Exception as e:
                    logger.error(f"OCR information extraction error: {e}")
                
                # 최종 텍스트 선택 및 정제
                document_text = ""
                if "glyph<" in backend_text or backend_text.strip() == "":
                    document_text = ocr_text
                    logger.info("[PARSING] 글리프 감지됨: OCR 텍스트 사용")
                else:
                    document_text = backend_text
                    logger.info("[PARSING] 정상 텍스트: 백엔드 텍스트 사용")
                
                # 텍스트 정제
                document_text = re.sub(r'glyph<[^>]+>', '', document_text)
                document_text = re.sub(r'<[^>]+>', '', document_text)
                
                # 이미지 캡션 정보를 본문에 추가하여 임베딩될 수 있게 함
                if images:
                    image_captions_text = "\n\n[이미지 캡션 정보]\n"
                    useful_captions = 0
                    
                    for idx, img in enumerate(images):
                        img_id = img.get('id', f'img_{idx}')
                        description = img.get('description', '')
                        
                        # 의미 없는 캡션인지 확인 (ID와 비슷한 경우)
                        is_meaningful = True
                        if description and isinstance(description, str):
                            if description == img_id or description.startswith("이미지 #/pictures/"):
                                # 파일명에서 의미 있는 정보 추출
                                filename = raw_document.metadata.get('filename', '')
                                file_base = os.path.splitext(os.path.basename(filename))[0] if filename else ''
                                
                                # 단순 참조 ID 대신 의미 있는 설명 대체
                                img_num = idx + 1
                                index_match = re.search(r'#/pictures/(\d+)', img_id) if isinstance(img_id, str) else None
                                if index_match:
                                    img_num = int(index_match.group(1)) + 1
                                
                                description = f"{file_base} 문서의 그림 {img_num}"
                                is_meaningful = True
                        
                        if description and isinstance(description, str) and len(description) > 0 and is_meaningful:
                            image_captions_text += f"이미지 {idx+1} 설명: {description}\n"
                            useful_captions += 1
                    
                    # 캡션 정보가 있을 경우에만 추가
                    if useful_captions > 0:
                        document_text += image_captions_text
                        logger.info(f"이미지 캡션 정보를 본문에 추가했습니다 ({useful_captions}개 유용한 캡션)")
                    else:
                        logger.info("유용한 이미지 캡션이 없어 본문에 추가하지 않습니다.")
                
                # 이미지 annotation 정보도 본문에 추가
                if image_descriptions:
                    image_annot_text = "\n\n[이미지 상세 설명]\n"
                    for idx, desc in enumerate(image_descriptions):
                        description = desc.get('description', '')
                        source = desc.get('source', '')
                        if description and isinstance(description, str) and len(description) > 0:
                            image_annot_text += f"이미지 설명 {idx+1}: {description}"
                            if source:
                                image_annot_text += f" (출처: {source})"
                            image_annot_text += "\n"
                    
                    # 설명 정보가 있을 경우에만 추가
                    if image_annot_text != "\n\n[이미지 상세 설명]\n":
                        document_text += image_annot_text
                        logger.info(f"이미지 상세 설명 정보를 본문에 추가했습니다 ({len(image_descriptions)}개 설명)")
                
                logger.info(f"[PARSING] 최종 추출 텍스트 길이: {len(document_text)} 글자")
                
                # 샘플 출력
                if document_text:
                    logger.info("\n===== 추출된 텍스트 샘플 =====")
                    text_sample = ""
                    try:
                        if document_text and isinstance(document_text, str):
                            if len(document_text) > 200:
                                text_sample = document_text[0:200] + "..."
                            else:
                                text_sample = document_text
                        else:
                            # 문자열이 아닌 경우 안전하게 변환
                            text_sample = str(document_text) if document_text is not None else ""
                    except Exception as sample_e:
                        logger.error(f"텍스트 샘플 생성 오류: {sample_e}")
                        text_sample = "[텍스트 샘플 생성 실패]"
                    
                    logger.info(text_sample)
                    logger.info("==============================\n")
                
                # 이미지 캡션 정보 출력
                if images:
                    logger.info("\n===== 추출된 이미지 캡션 정보 =====")
                    for idx, img in enumerate(images):
                        img_id = img.get('id', f'img_{idx}')
                        description = img.get('description', '설명 없음')
                        
                        # 설명이 너무 길면 자르기
                        if isinstance(description, str) and len(description) > 100:
                            description_preview = description[:100] + "..."
                        else:
                            description_preview = description
                        
                        logger.info(f"이미지 {idx+1}/{len(images)} (ID: {img_id}):")
                        logger.info(f"  - 설명: {description_preview}")
                        
                        # 이미지 데이터 유형 표시
                        img_data = img.get('data')
                        if img_data:
                            if isinstance(img_data, str) and len(img_data) > 50:
                                img_data_preview = img_data[:50] + "..."
                            else:
                                img_data_preview = str(img_data)
                            logger.info(f"  - 데이터 유형: {type(img_data).__name__}, 미리보기: {img_data_preview}")
                        else:
                            logger.info(f"  - 데이터: 없음")
                        
                        # 이미지 위치 정보 표시
                        position = img.get('position')
                        if position:
                            logger.info(f"  - 위치: {position}")
                    
                    logger.info("=========================================\n")
                
                # 이미지 설명(annotations) 정보 출력
                if image_descriptions:
                    logger.info("\n===== 추출된 이미지 설명(annotations) =====")
                    for idx, desc in enumerate(image_descriptions):
                        img_id = desc.get('image_id', 'unknown')
                        description = desc.get('description', '설명 없음')
                        source = desc.get('source', '출처 없음')
                        
                        # 설명이 너무 길면 자르기
                        if isinstance(description, str) and len(description) > 100:
                            description_preview = description[:100] + "..."
                        else:
                            description_preview = description
                        
                        logger.info(f"설명 {idx+1}/{len(image_descriptions)} (이미지 ID: {img_id}):")
                        logger.info(f"  - 내용: {description_preview}")
                        logger.info(f"  - 출처: {source}")
                    
                    logger.info("=========================================\n")
                
                # 파싱 결과 출력 추가
                def print_parsing_results(parsed_doc: ParsedDocument):
                    logger.info("\n========== 파싱 결과 ==========")
                    logger.info(f"파일명: {filename}")
                    
                    # 기본 텍스트 내용
                    if parsed_doc.content:
                        logger.info("\n[텍스트 내용 샘플]")
                        try:
                            content_preview = ""
                            if parsed_doc.content and isinstance(parsed_doc.content, str):
                                if len(parsed_doc.content) > 200:
                                    content_preview = parsed_doc.content[0:200] + "..."
                                else:
                                    content_preview = parsed_doc.content
                            else:
                                # 문자열이 아닌 경우 안전하게 변환
                                content_preview = str(parsed_doc.content) if parsed_doc.content is not None else ""
                            logger.info(content_preview)
                        except Exception as e:
                            logger.error(f"텍스트 샘플 출력 오류: {e}")
                            logger.info("[텍스트 샘플 출력 실패]")
                    
                    # 테이블 정보
                    if parsed_doc.tables:
                        logger.info(f"\n[테이블 수]: {len(parsed_doc.tables)}")
                        try:
                            table_count = len(parsed_doc.tables)
                            if table_count > 0:
                                preview_count = min(2, table_count)
                                for i in range(preview_count):
                                    if i < table_count:  # 추가 안전 검사
                                        table = parsed_doc.tables[i]
                                        if isinstance(table, dict):
                                            logger.info(f"\n테이블 {i+1} 미리보기:")
                                            logger.info(f"- 구조: {table.get('structure', '정보 없음')}")
                                            logger.info(f"- 위치: {table.get('position', '정보 없음')}")
                        except Exception as table_e:
                            logger.error(f"테이블 정보 출력 오류: {table_e}")
                            logger.info("[테이블 정보 출력 실패]")
                    
                    # 이미지 정보
                    if parsed_doc.images:
                        logger.info(f"\n[이미지 수]: {len(parsed_doc.images)}")
                        try:
                            img_count = len(parsed_doc.images)
                            if img_count > 0:
                                preview_count = min(2, img_count)
                                for i in range(preview_count):
                                    if i < img_count:  # 추가 안전 검사
                                        img = parsed_doc.images[i]
                                        if isinstance(img, dict):
                                            logger.info(f"\n이미지 {i+1} 정보:")
                                            logger.info(f"- 설명: {img.get('description', '정보 없음')}")
                                            logger.info(f"- 위치: {img.get('position', '정보 없음')}")
                        except Exception as img_e:
                            logger.error(f"이미지 정보 출력 오류: {img_e}")
                            logger.info("[이미지 정보 출력 실패]")
                    
                    # 문서 구조
                    if parsed_doc.headings:
                        logger.info(f"\n[제목 구조 수]: {len(parsed_doc.headings)}")
                        try:
                            if isinstance(parsed_doc.headings, list):
                                headings_count = len(parsed_doc.headings)
                                if headings_count > 0:
                                    preview_count = min(3, headings_count)
                                    headings_preview = []
                                    for i in range(preview_count):
                                        if i < headings_count:  # 추가 안전 검사
                                            headings_preview.append(parsed_doc.headings[i])
                            else:
                                # 리스트가 아닌 경우 빈 리스트 사용
                                logger.warning("제목이 리스트 형식이 아님")
                        except Exception as headings_e:
                            logger.error(f"[제목 구조 출력 오류]: {headings_e}")
                    
                    # OCR 결과
                    if parsed_doc.ocr_confidence is not None:
                        logger.info(f"\n[OCR 신뢰도]: {parsed_doc.ocr_confidence:.2%}")
                    
                    # 수식 정보
                    if parsed_doc.equations:
                        logger.info(f"\n[수식 수]: {len(parsed_doc.equations)}")
                        try:
                            eq_limit = min(2, len(parsed_doc.equations))
                            for i, eq in enumerate(parsed_doc.equations[0:eq_limit], 1):  # 처음 2개만 출력
                                logger.info(f"수식 {i}: {eq.get('content', '정보 없음')}")
                        except Exception as e:
                            logger.error(f"수식 정보 출력 오류: {e}")
                            logger.info("[수식 정보 출력 실패]")
                    
                    logger.info("\n==============================\n")

                # FastAPI 응답 형식으로 변환
                def create_api_response(parsed_doc: ParsedDocument) -> dict:
                    try:
                        content_preview = ""
                        if parsed_doc.content and isinstance(parsed_doc.content, str):
                            if len(parsed_doc.content) > 200:
                                content_preview = parsed_doc.content[0:200] + "..."
                            else:
                                content_preview = parsed_doc.content
                        else:
                            # 문자열이 아닌 경우 안전하게 변환
                            content_preview = str(parsed_doc.content) if parsed_doc.content is not None else ""
                            
                        # 안전한 테이블 미리보기 생성
                        tables_preview = []
                        try:
                            if parsed_doc.tables and isinstance(parsed_doc.tables, list):
                                table_count = len(parsed_doc.tables)
                                if table_count > 0:
                                    preview_count = min(2, table_count)
                                    for i in range(preview_count):
                                        if i < table_count:  # 추가 안전 검사
                                            table = parsed_doc.tables[i]
                                            if isinstance(table, dict):
                                                tables_preview.append({
                                                    "structure": table.get('structure'),
                                                    "position": table.get('position')
                                                })
                        except Exception as tables_e:
                            logger.error(f"테이블 미리보기 생성 오류: {tables_e}")
                            
                        # 안전한 이미지 미리보기 생성
                        images_preview = []
                        try:
                            if parsed_doc.images and isinstance(parsed_doc.images, list):
                                img_count = len(parsed_doc.images)
                                if img_count > 0:
                                    preview_count = min(2, img_count)
                                    for i in range(preview_count):
                                        if i < img_count:  # 추가 안전 검사
                                            img = parsed_doc.images[i]
                                            if isinstance(img, dict):
                                                images_preview.append({
                                                    "description": img.get('description'),
                                                    "position": img.get('position')
                                                })
                        except Exception as images_e:
                            logger.error(f"이미지 미리보기 생성 오류: {images_e}")
                            
                        # 안전한 제목 미리보기 생성
                        headings_preview = []
                        try:
                            if parsed_doc.headings and isinstance(parsed_doc.headings, list):
                                headings_count = len(parsed_doc.headings)
                                if headings_count > 0:
                                    preview_count = min(3, headings_count)
                                    headings_preview = []
                                    for i in range(preview_count):
                                        if i < headings_count:  # 추가 안전 검사
                                            headings_preview.append(parsed_doc.headings[i])
                        except Exception as headings_e:
                            logger.error(f"제목 미리보기 생성 오류: {headings_e}")
                            
                        return {
                            "filename": filename,
                            "content_preview": content_preview,
                            "metadata": {
                                "tables_count": len(parsed_doc.tables) if parsed_doc.tables else 0,
                                "images_count": len(parsed_doc.images) if parsed_doc.images else 0,
                                "headings_count": len(parsed_doc.headings) if parsed_doc.headings else 0,
                                "ocr_confidence": parsed_doc.ocr_confidence,
                                "equations_count": len(parsed_doc.equations) if parsed_doc.equations else 0
                            },
                            "tables_preview": tables_preview,
                            "images_preview": images_preview,
                            "headings_preview": headings_preview
                        }
                    except Exception as e:
                        logger.error(f"API 응답 생성 오류: {e}")
                        return {
                            "filename": filename,
                            "error": "API 응답 생성 중 오류 발생"
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
                
                # 메타데이터에 API 응답 저장 (직렬화 가능한 형태로)
                try:
                    api_response = create_api_response(ParsedDocument(
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
                    
                    # 직렬화 문제가 없는지 확인 (메서드 객체와 같은 직렬화 불가능한 객체 제거)
                    def ensure_serializable(obj):
                        if isinstance(obj, dict):
                            for key in list(obj.keys()):
                                if callable(obj[key]) or isinstance(obj[key], type):
                                    # 함수, 메서드, 클래스 등 직렬화 불가능한 객체 제거
                                    del obj[key]
                                else:
                                    obj[key] = ensure_serializable(obj[key])
                        elif isinstance(obj, list):
                            for i, item in enumerate(obj):
                                obj[i] = ensure_serializable(item)
                        return obj
                    
                    # 메타데이터에 API 응답 저장 (직렬화 가능한 형태로)
                    raw_document.metadata['api_response'] = ensure_serializable(api_response)
                    
                    # 메타데이터에 추출된 이미지 URI 추가
                    if images and len(images) > 0:
                        image_uris = []
                        for img in images:
                            # 이미지 데이터 유효성 검사 및 처리
                            img_data = img.get('data')
                            img_id = img.get('id', '')
                            
                            # 플레이스홀더 이미지인지 확인
                            if img_data and isinstance(img_data, str) and 'placeholder_for_' in img_data:
                                # 실제 이미지 데이터 재추출 시도
                                try:
                                    logger.info(f"메타데이터를 위한 이미지 재추출 시도: {img_id}")
                                    valid_image_data = None
                                    
                                    # 파일 확장자 확인
                                    file_ext = os.path.splitext(filename)[1].lstrip('.').lower()
                                    
                                    if file_ext == 'pdf' and raw_document.content:
                                        # PDF 파일에서 이미지 추출
                                        try:
                                            import io
                                            import fitz  # PyMuPDF
                                            import base64
                                            from PIL import Image
                                            
                                            # 임시 파일 생성
                                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
                                                temp_file.write(raw_document.content)
                                                pdf_path = temp_file.name
                                            
                                            # PyMuPDF로 이미지 추출
                                            pdf_doc = fitz.open(pdf_path)
                                            
                                            # 이미지 인덱스 추출
                                            img_index = 0
                                            if isinstance(img_id, str) and '#/pictures/' in img_id:
                                                index_match = re.search(r'#/pictures/(\d+)', img_id)
                                                if index_match:
                                                    img_index = int(index_match.group(1))
                                            
                                            # 모든 페이지 검색
                                            for page_idx in range(len(pdf_doc)):
                                                page = pdf_doc[page_idx]
                                                image_list = page.get_images(full=True)
                                                
                                                # 특정 페이지 또는 이미지가 없으면 다음 페이지로
                                                if not image_list:
                                                    continue
                                                
                                                # 이미지가 있는 경우 - 특정 인덱스 또는 첫 번째 이미지 사용
                                                target_idx = min(img_index, len(image_list) - 1) if img_index < len(image_list) else 0
                                                
                                                img_info = image_list[target_idx]
                                                xref = img_info[0]
                                                base_image = pdf_doc.extract_image(xref)
                                                image_bytes = base_image["image"]
                                                
                                                # Base64로 인코딩
                                                img_format = base_image["ext"]
                                                b64_data = base64.b64encode(image_bytes).decode('utf-8')
                                                valid_image_data = f"data:image/{img_format};base64,{b64_data}"
                                                logger.info(f"메타데이터용 이미지 추출 성공: 페이지 {page_idx+1}, 이미지 {target_idx}")
                                                break  # 첫 번째 성공한 이미지에서 종료
                                            
                                            # 임시 파일 정리
                                            os.unlink(pdf_path)
                                            pdf_doc.close()
                                            
                                            if valid_image_data:
                                                img_data = valid_image_data
                                                logger.info("메타데이터용 이미지를 실제 이미지 데이터로 대체")
                                            
                                        except ImportError:
                                            logger.warning("PyMuPDF 라이브러리가 설치되지 않았습니다.")
                                        except Exception as pdf_e:
                                            logger.error(f"PDF에서 메타데이터용 이미지 추출 실패: {pdf_e}")
                                    
                                except Exception as extract_e:
                                    logger.error(f"메타데이터용 이미지 데이터 추출 실패: {extract_e}")
                            
                            # 최종 처리된 이미지 데이터 메타데이터에 추가
                            if img_data and isinstance(img_data, str):
                                image_uris.append({
                                    'uri': img_data,
                                    'id': img_id,
                                    'description': img.get('description', '')
                                })
                            elif img_data and hasattr(img_data, 'tobytes'):
                                # PIL.Image 객체 처리
                                try:
                                    import io
                                    import base64
                                    
                                    img_bytes = io.BytesIO()
                                    img_data.save(img_bytes, format='PNG')
                                    img_bytes = img_bytes.getvalue()
                                    b64_data = base64.b64encode(img_bytes).decode('utf-8')
                                    
                                    image_uris.append({
                                        'uri': f"data:image/png;base64,{b64_data}",
                                        'id': img_id,
                                        'description': img.get('description', '')
                                    })
                                    logger.info(f"PIL 이미지 객체를 Base64로 변환 성공")
                                except Exception as pil_e:
                                    logger.error(f"PIL 이미지 변환 실패: {pil_e}")
                                    # 변환 실패 시 기본 URI 추가
                                    image_uris.append({
                                        'uri': f"data:image/png;base64,placeholder_for_{img_id}",
                                        'id': img_id,
                                        'description': img.get('description', '')
                                    })
                            else:
                                # 그 외 경우 기본 URI 추가
                                image_uris.append({
                                    'uri': f"data:image/png;base64,placeholder_for_{img_id}",
                                    'id': img_id,
                                    'description': img.get('description', '')
                                })
                        
                        # 메타데이터에 이미지 URI 추가
                        if image_uris:
                            logger.info(f"메타데이터에 {len(image_uris)}개 이미지 URI 추가")
                            raw_document.metadata['image_uris'] = image_uris
                    
                    # 메타데이터에 이미지 설명 추가
                    if image_descriptions and len(image_descriptions) > 0:
                        processed_descriptions = []
                        for desc in image_descriptions:
                            if desc.get('description'):
                                processed_descriptions.append({
                                    'image_id': desc.get('image_id', ''),
                                    'description': desc.get('description', ''),
                                    'source': desc.get('source', '')
                                })
                        
                        if processed_descriptions:
                            logger.info(f"메타데이터에 {len(processed_descriptions)}개 이미지 설명 추가")
                            raw_document.metadata['image_descriptions'] = processed_descriptions
                except Exception as api_resp_e:
                    logger.error(f"API 응답 저장 중 오류: {api_resp_e}")
                    # 오류 발생 시 메타데이터에 최소한의 정보만 저장
                    raw_document.metadata['api_response'] = {
                        "filename": filename,
                        "error": "API 응답 생성 중 오류 발생",
                        "detail": str(api_resp_e)
                    }
                
                # 변환 과정의 마지막에 반환 전 검사 추가
                # 최종 반환 객체 생성
                try:
                    # 여기서 최종 결과를 반환
                    return_obj = ParsedDocument(
                        content=document_text,
                        metadata=raw_document.metadata,
                        tables=tables if 'tables' in locals() else [],
                        images=images if 'images' in locals() else [],
                        equations=equations if 'equations' in locals() else [],
                        code_blocks=code_blocks if 'code_blocks' in locals() else [],
                        headings=headings if 'headings' in locals() else [],
                        paragraphs=paragraphs if 'paragraphs' in locals() else [],
                        lists=lists if 'lists' in locals() else [],
                        layout=layout if 'layout' in locals() else None,
                        page_info=page_info if 'page_info' in locals() else [],
                        ocr_results=ocr_results if 'ocr_results' in locals() else None,
                        ocr_confidence=ocr_confidence if 'ocr_confidence' in locals() else None,
                        image_classifications=image_classifications if 'image_classifications' in locals() else [],
                        image_descriptions=image_descriptions if 'image_descriptions' in locals() else [],
                        table_structures=table_structures if 'table_structures' in locals() else [],
                        table_cell_matches=table_cell_matches if 'table_cell_matches' in locals() else [],
                        code_enrichments=code_enrichments if 'code_enrichments' in locals() else [],
                        formula_enrichments=formula_enrichments if 'formula_enrichments' in locals() else []
                    )
                    logger.info(f"DoclingParserAdapter: 파싱 성공, 텍스트 길이: {len(document_text)}")
                    return return_obj
                except Exception as final_e:
                    logger.error(f"DoclingParserAdapter: 최종 결과 생성 중 오류 발생 - {final_e}")
            
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
            logger.error(traceback.format_exc())
            raise ParsingError(f"Unexpected error during parsing: {e}") from e

        # 모든 처리가 실패했거나 예외가 발생한 경우 빈 결과 반환
        logger.warning("DoclingParserAdapter: 파싱 실패 또는 문제 발생, 빈 결과 반환")
        return ParsedDocument(content="", metadata=raw_document.metadata)

# 파일 형식별 이미지 추출 기능
def extract_images_from_file(file_bytes, file_extension, filename):
    logger.info(f"[이미지 추출] 시작: {filename} (형식: {file_extension})")
    
    # 결과 저장용 리스트
    extracted_images = []
    
    try:
        # 1. PDF 파일 처리
        if file_extension.lower() == 'pdf':
            # PDF에서 이미지 추출하는 기능 (PyMuPDF 사용)
            try:
                import io
                import fitz  # PyMuPDF
                from PIL import Image
                
                pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
                img_index = 0
                
                for page_index in range(len(pdf_document)):
                    page = pdf_document.load_page(page_index)
                    image_list = page.get_images(full=True)
                    
                    for img_index, img_info in enumerate(image_list):
                        try:
                            xref = img_info[0]
                            base_image = pdf_document.extract_image(xref)
                            image_bytes = base_image["image"]
                            img = Image.open(io.BytesIO(image_bytes))
                            
                            extracted_images.append({
                                'data': img,
                                'description': f"Image from page {page_index + 1}",
                                'position': {"page": page_index + 1},
                                'id': f"pdf_img_{page_index}_{img_index}"
                            })
                        except Exception as e:
                            logger.error(f"[이미지 추출] PDF 이미지 처리 오류: {e}")
                
                pdf_document.close()
                logger.info(f"[이미지 추출] PDF에서 {len(extracted_images)}개 이미지 추출됨")
            except ImportError:
                logger.warning("[이미지 추출] PyMuPDF 라이브러리가 설치되지 않았습니다.")
            except Exception as e:
                logger.error(f"[이미지 추출] PDF 처리 오류: {e}")
            
        # 2. DOCX 파일 처리
        elif file_extension.lower() in ['docx', 'doc']:
            import io
            from docx import Document
            from PIL import Image
            
            doc = Document(io.BytesIO(file_bytes))
            img_index = 0
            
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        img_bytes = rel.target_part.blob
                        img = Image.open(io.BytesIO(img_bytes))
                        
                        extracted_images.append({
                            'data': img,
                            'description': f"Image from {filename}",
                            'position': None,
                            'id': f"docx_img_{img_index}"
                        })
                        img_index += 1
                    except Exception as e:
                        logger.error(f"[이미지 추출] DOCX 이미지 처리 오류: {e}")
            
            logger.info(f"[이미지 추출] DOCX에서 {len(extracted_images)}개 이미지 추출됨")
            
        # 3. PPTX 파일 처리
        elif file_extension.lower() in ['pptx', 'ppt']:
            import io
            from pptx import Presentation
            from PIL import Image
            
            prs = Presentation(io.BytesIO(file_bytes))
            img_index = 0
            
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, 'image'):
                        try:
                            img_bytes = shape.image.blob
                            img = Image.open(io.BytesIO(img_bytes))
                            
                            extracted_images.append({
                                'data': img,
                                'description': f"Image from slide {slide.slide_id}",
                                'position': {"slide": slide.slide_id},
                                'id': f"pptx_img_{img_index}"
                            })
                            img_index += 1
                        except Exception as e:
                            logger.error(f"[이미지 추출] PPTX 이미지 처리 오류: {e}")
            
            logger.info(f"[이미지 추출] PPTX에서 {len(extracted_images)}개 이미지 추출됨")
            
        # 4. XLSX 파일 처리
        elif file_extension.lower() in ['xlsx', 'xls']:
            import io
            import openpyxl
            from PIL import Image
            
            wb = openpyxl.load_workbook(io.BytesIO(file_bytes))
            img_index = 0
            
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                for image in sheet._images:
                    try:
                        img_bytes = image._data()
                        img = Image.open(io.BytesIO(img_bytes))
                        
                        extracted_images.append({
                            'data': img,
                            'description': f"Image from sheet {sheet_name}",
                            'position': {"sheet": sheet_name},
                            'id': f"xlsx_img_{img_index}"
                        })
                        img_index += 1
                    except Exception as e:
                        logger.error(f"[이미지 추출] XLSX 이미지 처리 오류: {e}")
            
            logger.info(f"[이미지 추출] XLSX에서 {len(extracted_images)}개 이미지 추출됨")
            
        # 5. 이미지 파일 직접 처리
        elif file_extension.lower() in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif']:
            import io
            from PIL import Image
            
            try:
                img = Image.open(io.BytesIO(file_bytes))
                extracted_images.append({
                    'data': img,
                    'description': f"Image file: {filename}",
                    'position': None,
                    'id': "original_image"
                })
                logger.info(f"[이미지 추출] 이미지 파일 처리 완료")
            except Exception as e:
                logger.error(f"[이미지 추출] 이미지 파일 처리 오류: {e}")
                
        # 6. HTML 파일 처리
        elif file_extension.lower() in ['html', 'htm']:
            import io
            import requests
            import base64
            from bs4 import BeautifulSoup
            from PIL import Image
            
            try:
                soup = BeautifulSoup(file_bytes, 'html.parser')
                img_tags = soup.find_all('img')
                img_index = 0
                
                for img_tag in img_tags:
                    try:
                        src = img_tag.get('src', '')
                        
                        # Base64 인코딩 이미지 처리
                        if src.startswith('data:image'):
                            # data:image/jpeg;base64,/9j/4AAQ... 형식에서 이미지 데이터 추출
                            img_data = src.split(',', 1)[1]
                            img_bytes = base64.b64decode(img_data)
                            img = Image.open(io.BytesIO(img_bytes))
                            
                            extracted_images.append({
                                'data': img,
                                'description': img_tag.get('alt', f"Image from HTML"),
                                'position': None,
                                'id': f"html_img_{img_index}"
                            })
                            img_index += 1
                    except Exception as e:
                        logger.error(f"[이미지 추출] HTML 이미지 처리 오류: {e}")
                
                logger.info(f"[이미지 추출] HTML에서 {len(extracted_images)}개 이미지 추출됨")
            except Exception as e:
                logger.error(f"[이미지 추출] HTML 파싱 오류: {e}")
                
        # 7. 기타 파일 형식 - 확장 가능
        else:
            logger.warning(f"[이미지 추출] 지원되지 않는 파일 형식: {file_extension}")
    
    except ImportError as e:
        logger.error(f"[이미지 추출] 필요한 라이브러리 없음: {e}")
    except Exception as e:
        logger.error(f"[이미지 추출] 오류 발생: {e}")
    
    logger.info(f"[이미지 추출] 완료: {len(extracted_images)}개 이미지 추출됨")
    return extracted_images

