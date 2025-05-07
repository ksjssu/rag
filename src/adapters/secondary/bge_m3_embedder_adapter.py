# src/adapters/secondary/bge_m3_embedder_adapter.py

# --- 임베딩 라이브러리 임포트 ---
# Sentence-Transformers 또는 Hugging Face transformers + torch 설치 필요:
# pip install sentence-transformers torch
# 또는 pip install transformers torch
try:
    # Sentence-Transformers 라이브러리 사용을 기본 예시로 합니다.
    from sentence_transformers import SentenceTransformer
    # 필요하다면 Hugging Face transformers 관련 클래스도 임포트 (fallback 또는 대체 구현 시)
    # from transformers import AutoTokenizer, AutoModel
    # import torch # PyTorch 필요

    _embedding_library_available = True
    print("Embedding library (sentence-transformers) imported successfully.")
except ImportError:
    print("Warning: Embedding library (sentence-transformers) not found. BgeM3EmbedderAdapter will use mock embeddings.")
    _embedding_library_available = False
    # --- 라이브러리가 없을 경우 에러 방지를 위한 더미 클래스 정의 ---
    class SentenceTransformer: # 더미 클래스
        def __init__(self, model_name, device=None, **kwargs):
            print(f"   (Simulating SentenceTransformer initialization - Library not available) Model: {model_name}, Device: {device}")
            # 더미 모델은 항상 로드 성공한다고 가정
            self._is_loaded = True
            self._model_name = model_name
            self._device = device
            self._mock_dimension = 1024 # BGE-M3 base 모델 차원 예시

        def encode(self, sentences, convert_to_list=False, **kwargs):
             print(f"   (Simulating SentenceTransformer.encode - Library not available) Encoding {len(sentences)} sentences...")
             import random # 모킹용
             # 결과는 List[List[float]] 형태로 시뮬레이션
             mock_embeddings = [[random.random() for _ in range(self._mock_dimension)] for _ in sentences]
             if not convert_to_list: # convert_to_list=False 시 numpy 배열 시뮬레이션
                 try:
                     import numpy as np
                     mock_embeddings = np.array(mock_embeddings)
                 except ImportError:
                      pass # numpy 없으면 그냥 list of lists 반환
             print(f"   (Simulating successful encode of {len(sentences)} embeddings)")
             return mock_embeddings

    # 필요하다면 Hugging Face 더미 클래스도 정의 (AutoTokenizer, AutoModel)

# --- 어댑터 특정 예외 정의 ---
# 임베딩 과정에서 발생하는 오류를 나타내기 위한 어댑터 레벨의 예외
class EmbeddingError(Exception):
    """Represents an error during the embedding generation process."""
    pass


import random # 모킹용으로 사용
from typing import List, Dict, Any, Optional # Optional 임포트

from ports.output_ports import EmbeddingGenerationPort, ApiKeyManagementPort # 구현할 포트 및 의존할 포트 임포트
from domain.models import DocumentChunk, EmbeddingVector # 입/출력 도메인 모델 임포트

# BGE-M3 모델은 보통 텍스트 임베딩 모델이므로, 텍스트 청크를 입력으로 받습니다.
# EmbeddingGenerationPort 인터페이스와 일치합니다.

class BgeM3EmbedderAdapter(EmbeddingGenerationPort):
    """
    BGE-M3 임베딩 모델을 사용하여 EmbeddingGenerationPort를 구현하는 어댑터.
    문서 청크 목록을 입력받아 해당 청크들의 임베딩 벡터 목록을 생성합니다.
    실제 BGE-M3 라이브러리 또는 API를 사용합니다.
    """
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3", # 사용할 모델 이름 (Sentence-Transformers 기준)
        device: Optional[str] = None, # 임베딩을 실행할 장치 (예: 'cuda', 'cpu', 'mps')
        api_key_port: Optional[ApiKeyManagementPort] = None, # 임베딩 서비스가 API 키 필요 시 주입
        # 임베딩 모델 초기화에 필요한 다른 파라미터 추가 (Docling 문서 확인)
        # 예: cache_dir: str = None # 모델 파일 캐싱 경로
    ):
        """
        BgeM3EmbedderAdapter 초기화 및 임베딩 모델 로드/클라이언트 설정.

        Args:
             model_name: 사용할 BGE-M3 모델의 식별자 (예: "BAAI/bge-m3").
                         로컬 경로 또는 Hugging Face 모델 ID.
             device: 임베딩을 실행할 장치 (예: 'cuda', 'cpu', 'mps'). None이면 자동 선택.
             api_key_port: API 키 관리가 필요한 경우 주입받는 포트 구현체 (클라우드 기반 API 사용 시).
        """
        print(f"BgeM3EmbedderAdapter: Initializing model '{model_name}' on device '{device or 'auto'}'...")

        self._model_name = model_name
        self._device = device
        self._api_key_port = api_key_port # API 키 포트 저장 (API 방식 임베더 사용 시)
        self._model: Optional[SentenceTransformer] = None # 임베딩 모델 인스턴스 저장 변수
        # self._tokenizer = None # transformers 사용 시 필요

        if not _embedding_library_available:
            print("BgeM3EmbedderAdapter: Embedding library not available. Will use mock embeddings.")
            # 라이브러리 없으면 더미 모델 인스턴스 생성
            # SentenceTransformer 더미 클래스가 __init__ 파라미터를 받도록 정의되어 있습니다.
            self._model = SentenceTransformer(self._model_name, device=self._device) # 더미 모델 인스턴스 생성
            if not hasattr(self._model, '_is_loaded') or not self._model._is_loaded: # 더미 모델 로드 상태 확인
                 print("BgeM3EmbedderAdapter: Mock model failed to simulate loading?") # 발생하지 않을 코드
                 raise EmbeddingError("Failed to initialize mock embedding model")
            print("BgeM3EmbedderAdapter: Mock embedding model initialized.")

        else: # 임베딩 라이브러리 사용 가능 시 실제 모델 로드 시도
            print(f"BgeM3EmbedderAdapter: Attempting to load model '{self._model_name}'...")
            try:
                # --- ★★★ 실제 임베딩 모델 로드 코드 ★★★ ---
                # 사용하는 라이브러리 (sentence-transformers 또는 transformers)에 따라 코드가 달라집니다.

                # --- Sentence-Transformers 사용 예시 ---
                # SentenceTransformer 클래스의 생성자 호출
                # model_name과 device 파라미터 전달
                # 캐싱 경로 등 다른 파라미터도 필요시 추가 (Sentence-Transformers 문서 확인)
                self._model = SentenceTransformer(
                    model_name=self._model_name,
                    device=self._device,
                    # cache_dir=... # 예시: 모델 파일 캐싱 경로
                )
                print("BgeM3EmbedderAdapter: Sentence-Transformers model loaded successfully.")

                # --- Hugging Face transformers 사용 예시 (주석 처리) ---
                # print(f"BgeM3EmbedderAdapter: Loading model '{self._model_name}' with transformers...")
                # # 토크나이저와 모델 분리 로드
                # self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                # # 모델 로드 후 장치 이동 (CUDA, CPU 등)
                # self._model = AutoModel.from_pretrained(self._model_name).to(self._device or ('cuda' if torch.cuda.is_available() else 'cpu'))
                # # 모델을 평가 모드로 설정 (추론 시 필요)
                # self._model.eval()
                # print("BgeM3EmbedderAdapter: Transformers model loaded successfully.")


            except Exception as e: # 모델 로드 중 발생할 수 있는 예외 처리
                print(f"BgeM3EmbedderAdapter: Error loading model '{self._model_name}': {e}")
                self._model = None # 로드 실패 시 모델 None
                # 모델 로드 실패 시 EmbeddingError 예외 발생시켜 앱 시작 중단 고려
                # raise EmbeddingError(f"Failed to load embedding model '{self._model_name}': {e}") from e


        # 초기화 성공 상태 확인
        if self._model is None:
            print("BgeM3EmbedderAdapter: Embedding model is not available or failed to load. Generate operations will use mock embeddings.")
        else:
             print("BgeM3EmbedderAdapter: Adapter successfully initialized with embedding model.")


    # Hugging Face transformers 사용 시 필요한 풀링 함수 예시 (sentence-transformers는 내장됨)
    # def _mean_pooling(self, model_output, attention_mask):
    #     token_embeddings = model_output[0]
    #     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    #     # attention mask를 적용하여 패딩 토큰의 임베딩이 합산되지 않도록 함
    #     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[EmbeddingVector]:
        """
        주어진 문서 청크 목록에 대해 BGE-M3 모델을 사용하여 임베딩 벡터를 생성합니다.
        """
        print(f"BgeM3EmbedderAdapter: Starting embedding generation for {len(chunks)} chunks...")

        if not chunks:
             print("BgeM3EmbedderAdapter: No chunks to embed. Skipping embedding generation.")
             return [] # 청크가 없으면 빈 목록 반환

        # --- 임베딩 생성 로직 (어댑터 내부) 또는 폴백 로직 실행 ---

        if self._model: # 임베딩 모델 인스턴스가 유효하면 실제 임베딩 실행
            print("BgeM3EmbedderAdapter: Using configured embedding model.")
            try:
                # --- 1단계: DocumentChunk 목록에서 텍스트 내용 목록 준비 ---
                chunk_contents = [chunk.content for chunk in chunks]
                print(f"   Encoding {len(chunk_contents)} text snippets...")

                # --- 2단계: API 키 조회 (임베딩 서비스가 API 방식인 경우) ---
                # 로컬 모델(Sentence-Transformers) 사용 시에는 필요 없습니다.
                # if self._api_key_port:
                #     try:
                #         api_key = self._api_key_port.get_api_key("BGE_M3_API_SERVICE") # 예시 서비스 이름
                #         # ... 임베딩 API 호출 시 api_key 사용 ...
                #         print("   (Retrieved API key for embedding service)")
                #     except ValueError as e:
                #          print(f"Warning: API key not found for embedding service: {e}. Cannot use API.")
                #          # API 키가 필수라면 여기서 EmbeddingError 발생 고려
                #          # raise EmbeddingError(f"API key required but not available: {e}") from e


                # --- ★★★ 3단계: 실제 임베딩 모델 추론(encode) 메서드 호출 ★★★ ---
                # 사용하는 라이브러리 (sentence-transformers 또는 transformers)에 따라 코드가 다릅니다.

                # --- Sentence-Transformers 사용 예시 ---
                # SentenceTransformer 모델의 encode 메서드 호출
                # 입력: 텍스트 내용 목록 (List[str])
                # convert_to_list=True로 설정하면 반환 결과를 파이썬 리스트 형태로 받습니다.
                embedding_output = self._model.encode( # <--- ▶︎▶︎▶︎ 실제 호출 라인! ◀︎◀︎◀︎
                    chunk_contents,
                    convert_to_list=True, # 결과를 파이썬 리스트 형태로 받기
                    show_progress_bar=False, # 진행률 바 숨김 (선택 사항)
                    # 기타 encode 메서드가 받는 파라미터 추가 (예: batch_size, normalize_embeddings)
                )
                embedding_results: List[List[float]] = embedding_output # 결과는 List[List[float]] 형태

                # --- Hugging Face transformers 사용 예시 (주석 처리) ---
                # # 텍스트를 토큰화하고 모델 입력 형태로 변환
                # inputs = self._tokenizer(chunk_contents, return_tensors='pt', padding=True, truncation=True).to(self._device)
                # # 모델 추론 실행 (requires torch)
                # with torch.no_grad(): # 임베딩 생성은 학습이 아니므로 no_grad 모드 사용
                #    model_output = self._model(**inputs)
                # # 모델 출력에서 임베딩 벡터 추출 및 풀링 적용
                # # 일반적으로 마지막 히든 스테이트의 평균 풀링을 사용합니다.
                # sentence_embeddings = self._mean_pooling(model_output, inputs['attention_mask']) # 위에서 정의한 풀링 함수 사용
                # # PyTorch 텐서를 파이썬 리스트로 변환
                # embedding_results: List[List[float]] = sentence_embeddings.tolist()
                # print("   (Using actual transformers encode and pooling)")


                # --- 4단계: 생성된 벡터 목록 처리 ---
                print(f"   Successfully generated {len(embedding_results)} raw vectors.")

                # 생성된 벡터 수와 청크 수가 일치하는지 확인 (불일치 시 오류 가능성)
                if len(chunks) != len(embedding_results):
                    print(f"Warning: Number of chunks ({len(chunks)}) and generated embeddings ({len(embedding_results)}) do not match.")
                    # 여기서 어떻게 처리할지 결정 (EmbeddingError 발생, 부분 결과 반환 등)
                    # 불일치 시 VectorDatabasePort.save_document_data에서 오류가 발생할 가능성이 높습니다.


            except Exception as e: # 임베딩 생성 중 발생할 수 있는 예외 처리
                 print(f"BgeM3EmbedderAdapter: Error during actual embedding generation - {e}")
                 # 임베딩 생성 중 발생한 예외를 어댑터 레벨의 EmbeddingError로 변환하여 다시 발생
                 raise EmbeddingError(f"Failed to generate embeddings: {e}") from e


        else: # self._model 인스턴스가 없거나 로드 실패 시 폴백 로직 실행
            print("BgeM3EmbedderAdapter: Using mock embedding generation (Model not available).")
            # ---> 모킹 벡터 생성 <---
            mock_dimension = 1024 # BGE-M3 base 모델 차원 예시 (실제 BGE-M3는 1024 차원일 수 있음)
            embedding_results = [[random.random() for _ in range(mock_dimension)] for _ in chunks]
            print(f"   Generated {len(embedding_results)} mock vectors.")


        # --- ★★★ 5단계: 생성된 벡터 목록을 EmbeddingVector 도메인 모델 목록으로 변환 ★★★ ---
        # 어댑터의 책임: 외부 기술 결과(벡터 리스트)를 내부 도메인 모델(EmbeddingVector 리스트)로 매핑
        embeddings: List[EmbeddingVector] = []
        try:
            print("   Mapping vectors to EmbeddingVector domain models...")
            # 생성된 벡터와 해당 청크 메타데이터를 조합하여 EmbeddingVector 객체 생성
            for i, vector in enumerate(embedding_results):
                # 해당 임베딩 벡터의 출처인 DocumentChunk의 메타데이터를 포함
                # 실제로는 청크 ID, 원본 문서 ID 등 임베딩을 검색 결과와 연결할 최소한의 메타데이터만 포함하는 것이 효율적일 수 있습니다.
                # 청크 목록과 임베딩 벡터 목록은 같은 순서라고 가정합니다.
                chunk_metadata_ref = chunks[i].metadata.copy() # 원본 청크 메타데이터 복사

                embeddings.append(EmbeddingVector(vector=vector, metadata=chunk_metadata_ref))

            print(f"BgeM3EmbedderAdapter: Mapping complete. Created {len(embeddings)} EmbeddingVector objects.")

        except Exception as e:
             print(f"BgeM3EmbedderAdapter: Error mapping vectors to domain models - {e}")
             # 매핑 중 오류 발생 시 EmbeddingError 발생
             raise EmbeddingError(f"Failed to map embedding results to domain models: {e}") from e


        print(f"BgeM3EmbedderAdapter: Embedding generation process finished. Generated {len(embeddings)} embeddings.")

        # 생성된 EmbeddingVector 객체 목록을 반환
        return embeddings