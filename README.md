# RAG Hex System

헥사고날 아키텍처를 적용한 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 프로젝트 구조

이 프로젝트는 헥사고날 아키텍처를 따르며 다음과 같은 계층으로 구성되어 있습니다:

- **Domain Layer**: 핵심 비즈니스 로직 및 엔티티
- **Application Layer**: 유스케이스
- **Ports Layer**: 인터페이스 정의
- **Adapters Layer**: 외부 시스템과의 통합

### 디렉토리 구조

```
src/
├── __init__.py
├── config.py               # 시스템 설정
├── main.py                 # 애플리케이션 진입점 및 의존성 주입
├── domain/                 # 도메인 레이어
│   ├── __init__.py
│   └── models.py           # 도메인 모델 (RawDocument, DocumentChunk, EmbeddingVector 등)
├── ports/                  # 포트 레이어
│   ├── __init__.py
│   ├── input_ports.py      # 입력 포트 정의 (DocumentProcessingInputPort)
│   └── output_ports.py     # 출력 포트 정의 (DocumentParsingPort, TextChunkingPort 등)
├── application/            # 애플리케이션 레이어
│   ├── __init__.py
│   └── use_cases.py        # 유스케이스 정의 (IngestDocumentUseCase)
└── adapters/               # 어댑터 레이어
    ├── __init__.py
    ├── primary/            # 프라이머리 어댑터 (시스템 외부에서 들어오는 요청 처리)
    │   ├── __init__.py
    │   └── api_adapter.py  # FastAPI 기반 REST API 어댑터
    └── secondary/          # 세컨더리 어댑터 (외부 시스템과의 통합)
        ├── __init__.py
        ├── docling_parser_adapter.py    # Docling 문서 파싱 어댑터
        ├── docling_chunker_adapter.py   # Docling 청킹 어댑터
        ├── bge_m3_embedder_adapter.py   # BGE-M3 임베딩 어댑터
        ├── milvus_adapter.py            # Milvus 벡터 데이터베이스 어댑터
        └── env_apikey_adapter.py        # API 키 관리 어댑터
```

## Docling 파이프라인

현재 시스템은 문서 처리를 위해 Docling 파이프라인을 사용하고 있습니다. 이는 다음과 같은 주요 기능을 제공합니다:

### 1. 문서 파싱 (DoclingParserAdapter)

다양한 문서 형식(PDF, DOCX, PPTX, 이미지 등)에서 텍스트 및 구조 정보를 추출합니다.

**주요 기능:**
- **텍스트 추출**: 문서에서 텍스트 콘텐츠 추출
- **OCR 처리**: PDF 및 이미지 파일에서 텍스트 인식 (한글 및 영어 지원)
- **레이아웃 분석**: 헤더, 푸터, 목록 등 문서 레이아웃 분석
- **테이블 분석**: 테이블 구조 인식 및 추출
- **이미지 처리**: 문서 내 이미지 추출 및 설명 생성
- **구조 분석**: 제목, 단락, 목록 등 문서 구조 분석

**PDF 파이프라인 설정:**
```python
pdf_options = PdfPipelineOptions(
    # OCR 설정
    do_ocr=True,
    ocr_options=EasyOcrOptions(
        lang=["ko", "en"],
        confidence_threshold=0.3,
        force_full_page_ocr=True
    ),
    
    # 테이블 처리
    do_table_structure=True,
    
    # 레이아웃 분석
    do_layout_analysis=True,
    
    # 이미지 처리
    do_picture_classification=True,
    do_picture_description=True,
    
    # 수식/코드 인식
    do_formula_enrichment=True,
    do_code_enrichment=True,
    
    # 하드웨어 설정
    accelerator_options=AcceleratorOptions(
        device="cpu",
        num_threads=4
    )
)
```

### 2. 문서 청킹 (DoclingChunkerAdapter)

파싱된 문서를 의미 있는 청크로 분할하여 임베딩 및 검색에 최적화합니다.

**주요 기능:**
- **하이브리드 청킹**: 구조 기반 + 고정 크기 청킹 방식 조합
- **청크 크기 제어**: 설정 가능한 청크 크기 및 오버랩
- **메타데이터 보존**: 소스, 이미지 설명, 테이블 정보 등 메타데이터 유지
- **특수 콘텐츠 처리**: 이미지, 테이블, 수식 등의 특수 콘텐츠 별도 청킹

### 3. 임베딩 생성 (BgeM3EmbedderAdapter)

청킹된 문서를 벡터로 변환하여 의미 기반 검색을 가능하게 합니다.

**주요 기능:**
- **BGE-M3 모델**: BAAI/bge-m3 모델 사용 (1024 차원 벡터)
- **하이브리드 임베딩**: Dense 임베딩 지원
- **배치 처리**: 효율적인 배치 처리 지원

### 4. 벡터 저장 (MilvusAdapter)

생성된 임베딩 벡터를 Milvus 벡터 데이터베이스에 저장합니다.

**주요 기능:**
- **벡터 저장**: 임베딩 벡터 및 메타데이터 저장
- **의미 검색**: 벡터 유사도 기반 검색 지원
- **메타데이터 검색**: 파일명, 청크 위치 등 메타데이터 기반 필터링

## 설치 방법

```bash
uv pip install .

실행 후 오류 뜰 시 아래 항목 추가 설치 (버전 기입 안해도 됨)
sentence-transformers
pymilvus[model]
docling
```

## 환경 설정

1. `.env` 파일을 생성하고 필요한 API 키를 설정합니다. // 들어가있는 키 삭제 예정
2. `config.py`에서 기본 설정을 확인하고 필요한 경우 수정합니다. // 들어가있는 기본 설정 삭제 예정

## 실행 방법

```bash
uvicorn src.main:app --reload
``` 

