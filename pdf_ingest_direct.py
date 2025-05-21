import os
import sys
import traceback
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 src 디렉터리를 파이썬 경로에 추가
sys.path.append(str(Path(__file__).parent / 'src'))

# 설정 및 로거 가져오기
from src.config import logger, settings

# 어댑터와 유스케이스 임포트
from docling.datamodel.pipeline_options import PdfPipelineOptions
from src.domain.models import RawDocument  # 추가: RawDocument 모델 임포트
from src.application.use_cases import IngestDocumentUseCase
from src.adapters.secondary.docling_parser_adapter import DoclingParserAdapter
from src.adapters.secondary.docling_chunker_adapter import DoclingChunkerAdapter
from src.adapters.secondary.bge_m3_embedder_adapter import BgeM3EmbedderAdapter
from src.adapters.secondary.milvus_adapter import MilvusAdapter, _milvus_library_available

# 더미 어댑터 정의
class DummyMilvusAdapter:
    def __init__(self, *args, **kwargs):
        logger.warning("Using DummyMilvusAdapter - Data will not be persisted")
    
    def save_document_data(self, chunks, embeddings):
        logger.warning("DummyMilvusAdapter: save_document_data called, but no action taken")
        return {"status": "dummy"}

# 파일을 RawDocument로 변환하는 함수 추가
def file_to_raw_document(file_path):
    """
    파일을 RawDocument 객체로 변환
    
    Args:
        file_path: 파일 경로
        
    Returns:
        RawDocument 객체
    """
    logger.info(f"파일을 RawDocument로 변환 중: {file_path}")
    try:
        # 파일 읽기
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # 메타데이터 생성
        filename = Path(file_path).name
        file_extension = Path(file_path).suffix.lower().lstrip('.')
        
        metadata = {
            'filename': filename,
            'path': str(file_path),
            'size': len(content),
            'type': f'application/{file_extension}' if file_extension else 'application/octet-stream'
        }
        
        # RawDocument 생성하여 반환
        return RawDocument(content=content, metadata=metadata)
    
    except Exception as e:
        logger.error(f"파일 변환 실패: {e}")
        raise

# 설정에 따른 어댑터 초기화 함수
def initialize_adapters():
    # 파이프라인 옵션 생성 (OCR, 테이블, 텍스트만)
    pdf_options = PdfPipelineOptions(
        do_ocr=settings.DO_OCR,
        do_table_structure=settings.DO_TABLE_STRUCTURE,
        images_scale=settings.IMAGES_SCALE,
        generate_picture_images=settings.GENERATE_PICTURE_IMAGES
    )

    # Parser Adapter
    parser_adapter = DoclingParserAdapter(
        allowed_formats=settings.DOCLING_ALLOWED_FORMATS.split(','),
        use_gpt_picture_description=False
    )
    # 청킹 Adapter
    chunker_adapter = DoclingChunkerAdapter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    # 임베딩 Adapter
    embedder_adapter = BgeM3EmbedderAdapter(
        model_name=settings.EMBEDDING_MODEL_NAME,
        device=settings.EMBEDDING_DEVICE
    )
    # Persistence Adapter (Milvus 또는 Dummy)
    if not _milvus_library_available:
        persistence_adapter = DummyMilvusAdapter()
    else:
        token = None
        if settings.MILVUS_USER and settings.MILVUS_PASSWORD:
            token = f"{settings.MILVUS_USER}:{settings.MILVUS_PASSWORD}"
        try:
            persistence_adapter = MilvusAdapter(
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
                collection_name=settings.MILVUS_COLLECTION,
                token=token
            )
        except Exception:
            persistence_adapter = DummyMilvusAdapter()
    return parser_adapter, chunker_adapter, embedder_adapter, persistence_adapter

# 메인 테스트 함수
def main():
    try:
        parser, chunker, embedder, persistence = initialize_adapters()
        # Ingest 유스케이스 생성
        ingest_uc = IngestDocumentUseCase(
            parser_port=parser,
            chunking_port=chunker,
            embedding_port=embedder,
            persistence_port=persistence
        )

        # raw 데이터가 위치한 디렉터리
        files_dir = Path(__file__).parent / 'pdffiles'
        for file_path in files_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() == '.pdf':
                logger.info(f"처리 중인 파일: {file_path.name}")
                try:
                    # 파일을 RawDocument로 변환
                    raw_document = file_to_raw_document(file_path)
                    
                    # 유스케이스의 execute 메서드 호출 (이전에는 존재하지 않는 ingest_document를 호출했음)
                    chunks = ingest_uc.execute(raw_document)
                    
                    logger.info(f"파일 처리 완료: {file_path.name}, 생성된 청크 수: {len(chunks)}")
                except Exception as e:
                    logger.error(f"파일 처리 오류 {file_path.name}: {e}")
                    traceback.print_exc()
    except Exception as e:
        logger.critical(f"테스트 파이프라인 실패: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
