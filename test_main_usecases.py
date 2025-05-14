import os
from pathlib import Path
from src.adapters.secondary.docling_parser_adapter import DoclingParserAdapter
from src.adapters.secondary.docling_chunker_adapter import DoclingChunkerAdapter
from src.adapters.secondary.bge_m3_embedder_adapter import BgeM3EmbedderAdapter
from src.adapters.secondary.env_apikey_adapter import EnvApiKeyAdapter
from src.adapters.secondary.milvus_adapter import MilvusAdapter
from src.application.use_cases import IngestDocumentUseCase
from src.domain.models import RawDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions # granite_picture_description

# 환경 변수/설정
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", 30953))
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "test_250430_1024_hybrid")
MILVUS_USER = os.getenv("MILVUS_USER", "root")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "smr0701!")
DOCLING_ALLOWED_FORMATS = os.getenv("DOCLING_ALLOWED_FORMATS", "pdf,docx,xlsx,pptx,jpg,png").split(',')
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", 1000))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 200))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# 어댑터 조립
apikey_adapter = EnvApiKeyAdapter()
parser_adapter = DoclingParserAdapter(
    allowed_formats=DOCLING_ALLOWED_FORMATS,
    use_gpt_picture_description=False
)
# DoclingParserAdapter 내부 파이프라인 옵션도 명시적으로 비활성화
if hasattr(parser_adapter, '_converter') and parser_adapter._converter is not None:
    pdf_format_option = parser_adapter._converter.format_to_options.get('pdf')
    if pdf_format_option and hasattr(pdf_format_option, 'pipeline_options'):
        pdf_format_option.pipeline_options.do_picture_description = False
chunker_adapter = DoclingChunkerAdapter(
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP
)
embedder_adapter = BgeM3EmbedderAdapter(
    model_name=EMBEDDING_MODEL_NAME,
    device=EMBEDDING_DEVICE
)
try:
    token = None
    if MILVUS_USER and MILVUS_PASSWORD:
        token = f"{MILVUS_USER}:{MILVUS_PASSWORD}"
    persistence_adapter = MilvusAdapter(
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        collection_name=MILVUS_COLLECTION,
        token=token
    )
except Exception:
    from src.main import DummyMilvusAdapter
    persistence_adapter = DummyMilvusAdapter()

ingest_use_case = IngestDocumentUseCase(
    parser_port=parser_adapter,
    chunking_port=chunker_adapter,
    embedding_port=embedder_adapter,
    persistence_port=persistence_adapter,
    api_key_port=apikey_adapter
)

# 테스트용 문서 경로 지정 (예: 샘플 PDF)
TEST_FILE = "./pdffiles/aaa.pdf"  # 실제 테스트할 파일 경로로 변경

if not Path(TEST_FILE).exists():
    print(f"테스트 파일이 존재하지 않습니다: {TEST_FILE}")
    exit(1)

with open(TEST_FILE, "rb") as f:
    content = f.read()

raw_doc = RawDocument(
    content=content,
    metadata={
        "filename": os.path.basename(TEST_FILE),
        "content_type": "application/pdf"
    }
)

# 유스케이스 실행
chunks = ingest_use_case.execute(raw_doc)
print(f"총 청크 개수: {len(chunks)}")
if chunks:
    print(f"첫 번째 청크 내용 일부: {chunks[0].content[:200]}")
    if hasattr(chunks[0], 'metadata'):
        print(f"첫 번째 청크 메타데이터: {chunks[0].metadata}")

# 이미지 설명 결과 예시 출력 (파싱 결과에서 추출)
parsed_doc = parser_adapter.parse(raw_doc)
breakpoint()
if hasattr(parsed_doc, 'image_descriptions'):
    print(f"이미지 설명 개수: {len(parsed_doc.image_descriptions)}")
    if parsed_doc.image_descriptions:
        print(f"첫 번째 이미지 설명: {parsed_doc.image_descriptions[0]}")
if hasattr(parsed_doc, 'tables'):
    print(f"테이블 개수: {len(parsed_doc.tables)}") 