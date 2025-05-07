# src/adapters/secondary/docling_parser_adapter.py

# --- Docling 라이브러리 임포트 ---
# 실제 설치하신 Docling 라이브러리의 정확한 임포트 구문을 사용하세요.
# 앞서 제공해주신 코드 기반으로 추정한 임포트입니다.
try:
    # Docling 파싱을 위한 핵심 클래스 임포트
    from docling_core.document_converter import DocumentConverter # DocumentConverter 클래스
    from docling_core.datamodel.base_models import DocumentStream, ConversionStatus, InputFormat # Docling 입력/결과 관련 모델
    from docling_core.datamodel.document import ConversionResult # Docling 결과 모델
    from docling_core.datamodel.pipeline_options import PipelineOptions, PdfPipelineOptions # 파이프라인 옵션 예시
    # 필요한 다른 Docling 모듈/클래스 임포트 (예외 클래스 등)
    from docling_core.exceptions import ConversionError as DoclingConversionError
    # Docling 자체 유틸리티 함수 임포트 (InputFormat 추정 등에 사용될 수 있음)
    # 예: from docling_core.utils.utils import guess_format_from_extension

    _docling_available = True
    print("Docling core libraries imported successfully.")
except ImportError:
    print("Warning: Docling core libraries not found (`docling_core`). DoclingParserAdapter will use fallback decoding.")
    _docling_available = False
    # --- Docling 클래스가 없을 경우 에러 방지를 위한 더미 클래스 정의 ---
    # 실제 Docling 클래스의 시그니처와 최대한 유사하게 정의하여 타입 힌트 에러를 줄입니다.
    print("   Using dummy Docling classes.")
    class DocumentConverter:
         def __init__(self, allowed_formats=None, format_options=None, **kwargs): pass
         def convert(self, source, headers=None, raises_on_error=True, **kwargs):
             print("   (Simulating DoclingConverter.convert - Library not available)")
             class MockDoc: # 더미 내부 문서 객체 (실제 Docling 내부 문서 객체 속성 반영 시도)
                  def __init__(self, text="", metadata=None): self._text = text; self._metadata = metadata or {}
                  def export_to_markdown(self): return self._text # PDFLoader 예시 반영
                  @property
                  def text(self): return self._text
                  @property
                  def metadata(self): return self._metadata or {} # None이 아닌 빈 dict 반환
                  def get_text(self): return self._text or ""
                  def get_metadata(self): return self._metadata or {}
             class MockConvResult: # 더미 ConversionResult
                 def __init__(self, status, document=None, errors=None, warnings=None):
                      self.status = status
                      self.document = document
                      self.errors = errors or []
                      self.warnings = warnings or []
             # Dummy ConversionStatus 사용
             _MockStatus = type("ConversionStatus", (object,), {'SUCCESS': 'SUCCESS', 'PARTIAL_SUCCESS': 'PARTIAL_SUCCESS', 'FAILURE': 'FAILURE', 'SKIPPED': 'SKIPPED'})
             # 입력 source에서 내용과 이름 추정하여 더미 문서 객체 생성
             content_str = ""
             name = "unknown"
             if isinstance(source, DocumentStream):
                  try: content_str = source.data.decode('utf-8', errors='ignore') # bytes->str 임시 변환
                  except: pass
                  name = source.name
             dummy_doc = MockDoc(text=f"Fallback content for {name}", metadata={'original_name': name})
             # 더미 에러/경고 메시지 추가
             errors = [f"Docling library not available or failed initialization. Input: {name}"] if not hasattr(self, '_is_initialized_successfully') or not self._is_initialized_successfully else []
             return MockConvResult(_MockStatus.FAILURE, document=None, errors=errors) # 라이브러리 없으면 무조건 실패 반환

    class DocumentStream: # 더미 클래스
        def __init__(self, data, name, format=None, **kwargs):
            self.data = data
            self.name = name
            self.format = format
    # 더미 InputFormat (Docling 코드 기반)
    class InputFormat:
        AUTODETECT = "AUTODETECT"; CSV="CSV"; XLSX="XLSX"; DOCX="DOCX"; PPTX="PPTX"; MD="MD"; ASCIIDOC="ASCIIDOC"; HTML="HTML"; XML_USPTO="XML_USPTO"; XML_JATS="XML_JATS"; IMAGE="IMAGE"; PDF="PDF"; JSON_DOCLING="JSON_DOCLING"
        __members__ = {name: type(name, (object,), {'name': name})() for name in [AUTODETECT, CSV, XLSX, DOCX, PPTX, MD, ASCIIDOC, HTML, XML_USPTO, XML_JATS, IMAGE, PDF, JSON_DOCLING]} # Enum 멤버처럼 보이도록
        @classmethod
        def from_extension(cls, ext): # 더미 from_extension
             ext_upper = ext.lstrip('.').upper()
             if ext_upper in cls.__members__: return cls.__members__[ext_upper]
             # 특별 매핑 처리 예시 (Docling 실제 코드 기반)
             if ext_upper in ['JPG', 'JPEG', 'PNG', 'TIFF'] and 'IMAGE' in cls.__members__: return cls.IMAGE
             return cls.AUTODETECT
        def __eq__(self, other): # 비교 가능하도록
             if isinstance(other, InputFormat): return self.name == other.name
             if isinstance(other, str): return self.name == other.upper() # 문자열과 비교 가능
             return False
        def __hash__(self): return hash(self.name) # 해시 가능하도록
        def __str__(self): return self.name # 문자열 표현
        def __repr__(self): return f"<InputFormat:{self.name}>"


    # 더미 ConversionResult (위에서 정의된 MockConvResult와 동일)
    class ConversionResult:
         def __init__(self, status, document=None, errors=None, warnings=None): self.status=status; self.document=document; self.errors=errors or []; self.warnings=warnings or []
         @property
         def status(self): return self._status
         @status.setter
         def status(self, value): self._status = value
         @property
         def document(self): return self._document
         @document.setter
         def document(self, value): self._document = value

    # 더미 ConversionStatus enum
    class ConversionStatus:
        SUCCESS = "SUCCESS"; PARTIAL_SUCCESS = "PARTIAL_SUCCESS"; FAILURE = "FAILURE"; SKIPPED = "SKIPPED"


    class DoclingConversionError(Exception): pass # 더미 예외


# --- 어댑터 특정 예외 정의 ---
# 파싱 과정에서 발생하는 오류를 나타내기 위한 어댑터 레벨의 예외
class ParsingError(Exception):
    """Represents an error during the document parsing process."""
    pass


import os
from ports.output_ports import DocumentParsingPort # 구현할 포트 임포트
from domain.models import RawDocument, ParsedDocument # 입/출력 도메인 모델 임포트
from typing import Dict, Any, Optional, List, Union # Union 임포트
from pathlib import Path # 파일 확장자 추출에 사용


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
            class DummyFormatOption: # FormatOption 더미
                 def __init__(self, pipeline_cls, backend, pipeline_options=None): pass
            class DummyPdfFormatOption(DummyFormatOption): # PdfFormatOption 더미
                 def __init__(self, pipeline_options=None): pass


            if self._pdf_options:
                 try:
                      # Docling InputFormat.PDF와 PdfFormatOption, PdfPipelineOptions 클래스가 필요합니다.
                      # Docling 문서에서 PdfFormatOption의 생성자 파라미터 확인 (pipeline_options를 받는지 등)
                      # format_options_dict[InputFormat.PDF] = PdfFormatOption(pipeline_options=self._pdf_options) # 실제 코드 형태
                      # 더미 클래스 사용 예시
                      if 'PDF' in InputFormat.__members__:
                          format_options_dict[InputFormat.PDF] = DummyPdfFormatOption(pipeline_options=self._pdf_options)
                          print("   (Added PDF options to format_options_dict using dummy classes)")
                 except Exception as e:
                      print(f"Warning: Could not configure PDF options for DoclingConverter: {e}")


        if _docling_available:
            self._is_initialized_successfully = False # 초기화 성공 여부 플래그
            print("DoclingParserAdapter: Initializing Docling DocumentConverter...")
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
                              print(f"Warning: Specified allowed_format '{fmt_str}' is not a valid Docling InputFormat.")


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
                 print("DoclingParserAdapter: Docling DocumentConverter initialized successfully.")
            except Exception as e: # Docling Converter 초기화 중 발생할 수 있는 예외 처리
                print(f"Error initializing Docling DocumentConverter: {e}")
                self._converter = None # 초기화 실패 시 None으로 설정
                # 초기화 실패 시 ParsingError 예외를 발생시켜 앱 시작 중단 고려
                # raise ParsingError(f"Failed to initialize Docling Converter: {e}") from e


        if self._converter is None:
             print("DoclingParserAdapter: Docling Converter not available or failed to initialize. Will use fallback decoding.")


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
                 print(f"Warning: Error guessing Docling InputFormat using utility: {e}. Falling back to manual guess.")
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
        print(f"DoclingParserAdapter: Parsing document {raw_document.metadata.get('filename', 'untitled')}...")

        parsed_content = ""
        extracted_metadata = raw_document.metadata.copy()
        docling_internal_document = None # Docling 파싱 결과 객체 (conversion_result.document) 저장 변수
        parsing_success = False # 파싱 성공 여부 플래그 (Docling 처리 결과 기준)


        # Docling Converter가 유효하고 초기화 성공했다면 실제 파싱 실행
        if self._converter is not None:
            print("DoclingParserAdapter: Using configured Docling Converter.")
            try:
                # --- 1단계: RawDocument의 내용을 Docling 라이브러리가 받는 입력 형식으로 변환 ---
                # 제공된 Docling 코드에 DocumentStream 클래스가 있고 convert 메서드가 받으므로 사용
                # DocumentStream 생성자/사용법은 Docling 문서를 확인해야 합니다.
                # data (bytes), name (filename), format (InputFormat enum) 등을 받을 수 있습니다.
                print("   Preparing input for Docling DocumentConverter...")
                doc_format = self._guess_input_format(raw_document.metadata) # 포맷 추정

                doc_stream = DocumentStream( # <--- Docling 라이브러리의 DocumentStream 클래스 사용
                     data=raw_document.content,
                     name=raw_document.metadata.get('filename', 'file'), # 파일명 전달 (Docling이 이것으로 형식 추정할 수 있음)
                     format=doc_format, # 추정한 포맷을 명시적으로 전달 (Docling 문서 확인, 자동 감지 우선 시 None)
                     # 기타 DocumentStream이 받는 파라미터 추가 (Docling 문서 확인)
                )
                print(f"   Prepared DocumentStream (name='{doc_stream.name}', format='{doc_stream.format}')")

                # --- ★★★ 2단계: 실제 Docling 라이브러리 파싱 기능을 호출하는 부분 ★★★ ---
                print("   Calling self._converter.convert()...")
                # self._converter는 __init__에서 생성한 DocumentConverter 인스턴스
                # .convert() 메서드는 Docling 라이브러리의 핵심 파싱 메서드
                # source 파라미터에 준비한 DocumentStream 객체를 전달
                # PDFLoader 예시처럼 headers 등 다른 파라미터도 전달 가능
                # raises_on_error=False로 설정하면 예외 대신 ConversionResult에 오류가 담겨 반환됩니다.
                # 어댑터가 오류를 명시적으로 처리하기 위해 False가 유리합니다.
                conversion_result: ConversionResult = self._converter.convert( # <--- ▶︎▶︎▶︎ 실제 호출 라인! ◀︎◀︎◀︎
                    source=doc_stream,
                    headers=raw_document.metadata.get('headers'),
                    raises_on_error=False, # 어댑터가 직접 결과를 보고 판단하도록 설정
                    # max_num_pages, max_file_size, page_range 등 (필요시 Docling 문서 확인)
                    # Docling convert 메서드가 받는 다른 파라미터 추가 (Docling 문서 확인)
                )
                # --- 호출 결과는 Docling의 ConversionResult 객체입니다. ---
                print(f"   Received ConversionResult with status: {conversion_result.status.name}")


                # --- 3단계: Docling 파싱 결과(ConversionResult)를 받아서 처리 ---
                print("   Processing Docling ConversionResult...")
                # 반환된 ConversionResult 객체의 상태를 확인하고 필요한 데이터(파싱된 텍스트, 메타데이터)를 추출합니다.

                if conversion_result.status in {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}:
                    parsing_success = True # 파싱 성공 (부분 성공 포함)
                    # 파싱 성공/부분 성공 시 Docling 내부 문서 객체 접근
                    docling_internal_document = conversion_result.document # <--- ConversionResult에서 문서 객체 얻기

                    if docling_internal_document:
                        # --- Docling 내부 문서 객체에서 필요한 데이터 추출 (Docling 문서 확인) ---
                        # 제공된 PDFLoader 예시에서 .export_to_markdown()을 사용하여 텍스트를 얻었음을 확인
                        # 다른 메타데이터는 .metadata 속성 등을 추정 (Docling 문서 확인 필요)
                        print("   Extracting text and metadata from Docling internal document...")
                        # 텍스트 추출: .export_to_markdown() 또는 .text 속성/get_text() 메서드
                        # Docling 문서에서 정확한 방법을 확인하세요. 우선순위 부여
                        extracted_text = ""
                        if hasattr(docling_internal_document, 'export_to_markdown'):
                             # PDFLoader 예제 기반: export_to_markdown 사용 (마크다운 형태)
                             extracted_text = docling_internal_document.export_to_markdown() or ""
                             print("   (Extracted text using export_to_markdown)")
                        elif hasattr(docling_internal_document, 'text') and isinstance(docling_internal_document.text, str):
                             # .text 속성이 있다면 사용 (일반 텍스트일 가능성)
                             extracted_text = docling_internal_document.text
                             print("   (Extracted text using .text attribute)")
                        elif hasattr(docling_internal_document, 'get_text'):
                             # get_text() 메서드가 있다면 사용
                             extracted_text = docling_internal_document.get_text() or ""
                             print("   (Extracted text using .get_text() method)")
                        else:
                             print("   (Docling document has no recognizable text extraction method/attribute)")


                        # 메타데이터 추출 (Docling 문서에서 Docling 내부 문서 객체의 속성/메서드 확인)
                        # 예: .metadata 속성, .get_metadata() 메서드 등
                        extracted_docling_metadata = {}
                        if hasattr(docling_internal_document, 'metadata') and isinstance(docling_internal_document.metadata, dict):
                              extracted_docling_metadata = docling_internal_document.metadata
                              print("   (Extracted metadata using .metadata attribute)")
                        elif hasattr(docling_internal_document, 'get_metadata'):
                              extracted_docling_metadata = docling_internal_document.get_metadata() or {}
                              print("   (Extracted metadata using .get_metadata())")
                        else:
                              print("   (Docling document has no .metadata attribute or .get_metadata() method)")

                        # 원본 메타데이터에 Docling에서 추출된 메타데이터 병합
                        # Docling 메타데이터가 원본 메타데이터를 덮어쓸 수 있습니다. 정책 결정 필요.
                        extracted_metadata.update(extracted_docling_metadata)
                        # Docling 결과에 포함된 오류/경고 정보도 메타데이터에 추가하는 것이 좋습니다.
                        if conversion_result.errors: extracted_metadata['docling_errors'] = [str(e) for e in conversion_result.errors]
                        if conversion_result.warnings: extracted_metadata['docling_warnings'] = [str(w) for w in conversion_result.warnings]

                        print("   Docling parsing successfully processed.")
                        parsed_content = extracted_text # 추출한 텍스트를 최종 parsed_content로 사용


                    else: # Status indicates success/partial success but document is None? (가능성은 낮으나 발생 시)
                         print(f"   Warning: Docling status {conversion_result.status.name} but document object is None.")
                         parsed_content = "" # 문서 객체가 없으면 내용도 없음
                         extracted_metadata['docling_status'] = conversion_result.status.name
                         if conversion_result.errors: extracted_metadata['docling_errors'] = [str(e) for e in conversion_result.errors]
                         # 여기서 파싱 실패로 간주하고 예외를 발생시킬지 결정
                         parsing_success = False


                else: # Docling 파싱 실패 상태 (ConversionStatus.FAILURE, SKIPPED 등)
                    parsing_success = False
                    print(f"   Docling parsing failed with status: {conversion_result.status.name}")
                    parsed_content = f"Docling parsing failed. Status: {conversion_result.status.name}"
                    extracted_metadata['docling_status'] = conversion_result.status.name
                    if conversion_result.errors:
                         extracted_metadata['docling_errors'] = [str(e) for e in conversion_result.errors]
                         parsed_content += f" Errors: {[str(e) for e in conversion_result.errors]}"
                    docling_internal_document = None # 실패했으므로 내부 객체 없음

                # Docling 파싱이 최종적으로 성공하지 않았다면 (status == FAILURE/SKIPPED 또는 document is None 등)
                # 어댑터 레벨의 ParsingError를 발생시켜 유스케이스에게 알립니다.
                if not parsing_success:
                     error_detail = extracted_metadata.get('docling_errors', [f"Status: {conversion_result.status.name}"])
                     error_message = f"Docling parsing failed for {raw_document.metadata.get('filename')}: {'. '.join(error_detail)}"
                     print(f"DoclingParserAdapter: Raising ParsingError: {error_message}")
                     raise ParsingError(error_message)


            except DoclingConversionError as e:
                 # Docling 라이브러리에서 정의한 특정 예외 처리
                 print(f"DoclingParserAdapter: Docling ConversionError occurred - {e}")
                 # 어댑터 레벨의 ParsingError로 변환하여 다시 발생
                 raise ParsingError(f"Docling conversion error for {raw_document.metadata.get('filename')}: {e}") from e
            except Exception as e:
                 # Docling 호출 중 발생할 수 있는 예상치 못한 다른 예외 처리
                 print(f"DoclingParserAdapter: An unexpected error occurred during Docling call - {e}")
                 # 어댑터 레벨의 ParsingError로 변환하여 다시 발생
                 raise ParsingError(f"Unexpected error during Docling parsing for {raw_document.metadata.get('filename')}: {e}") from e


        else: # self._converter가 None인 경우 (Docling 라이브러리 임포트/초기화 실패 등)
            # Docling 라이브러리가 없거나 사용 불가능할 때의 폴백 로직
            print("DoclingParserAdapter: Using fallback simple decoding (Docling Converter not available or failed to initialize).")
            try:
                parsed_content = raw_document.content.decode('utf-8', errors='replace')
                # 폴백 사용 시 Docling 내부 객체는 당연히 없음
                docling_internal_document = None
                parsing_success = True # 폴백은 일단 성공으로 간주
                print("   Fallback decoding successful.")
            except Exception as e:
                print(f"DoclingParserAdapter: Fallback decoding failed - {e}")
                parsed_content = f"Error decoding document content using fallback: {e}"
                docling_internal_document = None
                parsing_success = False # 폴백 실패
                # 폴백마저 실패 시 ParsingError를 발생시킬지 결정
                raise ParsingError(f"Fallback parsing failed for {raw_document.metadata.get('filename')}: {e}") from e


        print("DoclingParserAdapter: Parsing process finished.")

        # --- ★★★ 4단계: 추출한 결과와 내부 객체를 ParsedDocument 도메인 모델에 담아 반환 ★★★ ---
        # 어댑터의 역할: 외부 기술 결과를 내부 도메인 모델로 변환하여 반환
        # 청킹 어댑터가 Docling 내부 문서 객체를 필요로 하므로 메타데이터에 담아서 전달합니다.
        # Docling 내부 객체가 None이더라도 일단 담아서 전달합니다. (청킹 어댑터에서 None 체크)
        # 원본 메타데이터를 기본으로 하고, Docling 파싱에서 얻은 메타데이터를 업데이트합니다.
        # Docling 내부 객체도 메타데이터에 추가합니다.
        final_metadata = raw_document.metadata.copy()
        final_metadata.update(extracted_metadata) # 파싱 중 추출/병합된 메타데이터 추가
        final_metadata['__internal_docling_document__'] = docling_internal_document # 내부 객체 추가


        return ParsedDocument(content=parsed_content, metadata=final_metadata)