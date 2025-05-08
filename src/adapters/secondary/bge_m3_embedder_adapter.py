# src/adapters/secondary/bge_m3_embedder_adapter.py

import logging
import random
from typing import List, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# --- 임베딩 라이브러리 임포트 ---
# Sentence-Transformers 또는 Hugging Face transformers + torch 설치 필요:
# pip install sentence-transformers torch
# 또는 pip install transformers torch
# src/adapters/secondary/bge_m3_embedder_adapter.py

# --- 임베딩 라이브러리 임포트 ---
# BGE-M3 임베딩 기능은 pymilvus 라이브러리에서 제공됩니다.
# sentence-transformers 또는 transformers 라이브러리는 더 이상 직접 필요 없습니다.
try:
    # BGEM3EmbeddingFunction 클래스는 pymilvus.model.hybrid에 있습니다.
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction # <-- pymilvus에서 임포트
    # pymilvus 라이브러리 임포트 여부 플래그 (MilvusAdapter.py에서 정의)
    # from .milvus_adapter import _milvus_library_available # <-- MilvusAdapter에서 플래그 가져오기 시도

    _embedding_library_available = True # BGEM3EmbeddingFunction 사용 가능 여부 (pymilvus 사용 가능 여부와 연결)
    print("BGEM3EmbeddingFunction from pymilvus imported successfully.")
except ImportError as e:
    print(f"Warning: BGEM3EmbeddingFunction from pymilvus not found. Import error: {e}") # <-- 오류 메시지 수정
    print("BgeM3EmbedderAdapter will use mock embeddings.")
    _embedding_library_available = False # BGEM3EmbeddingFunction 사용 불가

    # --- 임베딩 라이브러리가 없을 경우 에러 방지를 위한 더미 클래스 정의 ---
    print("   Using dummy embedding classes.")
    # Dummy class for BGEM3EmbeddingFunction (minimal implementation)
    class BGEM3EmbeddingFunction: # <-- 더미 클래스 정의
        # 제공된 코드 기반 생성자 시그니처 반영
        def __init__(self, model_name='dummy', device=None, use_fp16=False):
             print(f"   (Simulating Dummy BGEM3EmbeddingFunction initialization - Library not available) Model: {model_name}, Device: {device}, use_fp16={use_fp16}")
             self._model_name = model_name
             self._device = device
             self._use_fp16 = use_fp16
             self._mock_dimension = 1024 # BGE-M3 차원 (1024로 추정)

        # encode 메서드 시뮬레이션
        def encode(self, sentences): # <-- encode 메서드는 list[str] 입력만 받음 (제공된 코드 기반)
             print(f"   (Simulating Dummy BGEM3EmbeddingFunction.encode) Encoding {len(sentences)} sentences...")
             import random # 모킹용
             # 결과는 List[List[float]] 형태로 시뮬레이션
             mock_embeddings = [[random.random() for _ in range(self._mock_dimension)] for _ in sentences]
             print(f"   (Simulating successful encode of {len(sentences)} embeddings)")
             return mock_embeddings

    # BaseTokenizer 더미 클래스는 DoclingChunkerAdapter에서 사용되므로 여기서는 필요 없습니다.
    # DoclingDocument 더미 클래스도 마찬가지입니다.


# --- 어댑터 특정 예외 정의 ---
class EmbeddingError(Exception):
    """Represents an error during the embedding generation process."""
    pass


import random # 모킹용
from typing import List, Dict, Any, Optional # Optional 임포트

from ports.output_ports import EmbeddingGenerationPort, ApiKeyManagementPort # 구현할 포트 및 의존할 수 있는 포트 임포트
from domain.models import DocumentChunk, EmbeddingVector # 입/출력 도메인 모델 임포트


# Dummy Embedder Adapter (pymilvus 부재 시 main.py에서 이 어댑터를 주입)
class DummyEmbedderAdapter(EmbeddingGenerationPort):
    """pymilvus 라이브러리 부재 시 사용되는 임베딩 어댑터 더미."""
    def __init__(self, model_name: str = "dummy-bge-m3", device: Optional[str] = "cpu"):
        print(f"DummyEmbedderAdapter initialized for model '{model_name}' on device '{device}'.")
        self._model_name = model_name
        self._device = device
        self._mock_dimension = 1024 # BGE-M3 차원 (1024로 추정)

    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[EmbeddingVector]:
        print(f"DummyEmbedderAdapter: Using mock embeddings for {len(chunks)} chunks...")
        if not chunks:
            return []
        # Mock embedding generation logic (same as before)
        mock_embeddings_list = [[random.random() for _ in range(self._mock_dimension)] for _ in chunks]
        embeddings: List[EmbeddingVector] = []
        for i, vector in enumerate(mock_embeddings_list):
            chunk_metadata_ref = chunks[i].metadata.copy()
            embeddings.append(EmbeddingVector(vector=vector, metadata=chunk_metadata_ref))
        print(f"DummyEmbedderAdapter: Generated {len(embeddings)} mock embeddings.")
        return embeddings


class BgeM3EmbedderAdapter(EmbeddingGenerationPort):
    """
    pymilvus 라이브러리의 BGEM3EmbeddingFunction을 사용하여 EmbeddingGenerationPort를 구현하는 어댑터.
    문서 청크 목록을 입력받아 해당 청크들의 임베딩 벡터 목록을 생성합니다.
    """
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3", # BGEM3EmbeddingFunction에 전달될 모델 이름
        device: Optional[str] = None, # BGEM3EmbeddingFunction에 전달될 장치
        use_fp16: bool = False, # BGEM3EmbeddingFunction에 전달될 FP16 사용 여부
        #api_key_port: Optional[ApiKeyManagementPort] = None, # BGEM3EmbeddingFunction은 토큰을 직접 받지 않는 것으로 추정
        # 기타 BGEM3EmbeddingFunction 초기화에 필요한 파라미터 추가 (pymilvus 문서 확인)
    ):
        """
        BgeM3EmbedderAdapter 초기화 및 BGEM3EmbeddingFunction 인스턴스 생성.

        Args:
             model_name: BGEM3EmbeddingFunction에 전달될 모델 이름.
             device: BGEM3EmbeddingFunction에 전달될 장치 ('cpu', 'cuda', 'etc').
             use_fp16: BGEM3EmbeddingFunction에 전달될 FP16 사용 여부. CPU 사용 시 False 권장.
             # 기타 BGEM3EmbeddingFunction 초기화 파라미터들 (pymilvus 문서 확인 필요)
        """
        print(f"BgeM3EmbedderAdapter: Initializing for model '{model_name}' on device '{device or 'auto'}' with use_fp16={use_fp16}...")

        self._model_name = model_name
        self._device = device
        self._use_fp16 = use_fp16
        # self._api_key_port = api_key_port # API 키 포트는 여기서 직접 사용되지 않는 것으로 보임 (BGEM3EmbeddingFunction이 내부적으로 처리 추정)

        self._embedding_function: Optional[BGEM3EmbeddingFunction] = None # BGEM3EmbeddingFunction 인스턴스

        # BGEM3EmbeddingFunction은 pymilvus 라이브러리에 포함되어 있으므로,
        # pymilvus 임포트 성공 여부(_milvus_library_available 플래그)를 확인하여 초기화합니다.
        # _milvus_library_available 플래그는 MilvusAdapter.py 파일에 정의되어 있습니다.
        # main.py에서 _milvus_library_available를 임포트하고 이 값을 사용하여
        # BgeM3EmbedderAdapter 또는 DummyEmbedderAdapter를 선택하도록 조립해야 합니다.
        # 이 어댑터 자체는 BGEM3EmbeddingFunction 클래스가 임포트 가능할 때만 초기화를 시도합니다.

        # --- ★★★ BGEM3EmbeddingFunction 인스턴스 생성 ★★★ ---
        # BGEM3EmbeddingFunction 클래스가 임포트 가능할 때만 이 코드가 실행됩니다.
        if 'BGEM3EmbeddingFunction' in globals() and isinstance(BGEM3EmbeddingFunction, type): # <-- 클래스가 정의되었는지 확인 (ImportError 안 났다는 의미)
             print(f"   Attempting to instantiate BGEM3EmbeddingFunction for model '{self._model_name}'...")
             try:
                 # 제공된 코드 스니펫의 BGEM3EmbeddingFunction 생성자 시그니처 사용
                 self._embedding_function = BGEM3EmbeddingFunction(
                     model_name=self._model_name,
                     device=self._device,
                     use_fp16=self._use_fp16,
                     # 기타 BGEM3EmbeddingFunction 생성자가 받는 파라미터 추가 (pymilvus 문서 확인)
                 )
                 print("BgeM3EmbedderAdapter: BGEM3EmbeddingFunction instance created successfully.")
             except Exception as e: # BGEM3EmbeddingFunction 초기화 중 발생할 수 있는 예외 처리
                 print(f"BgeM3EmbedderAdapter: Error initializing BGEM3EmbeddingFunction: {e}")
                 self._embedding_function = None # 초기화 실패 시 None
                 # 초기화 실패 시 EmbeddingError 예외 발생 고려
                 # raise EmbeddingError(f"Failed to initialize BGEM3EmbeddingFunction: {e}") from e
        else:
             print("BgeM3EmbedderAdapter: BGEM3EmbeddingFunction class not available. Likely pymilvus not fully imported.")
             self._embedding_function = None # 클래스 자체가 없으면 None


        if self._embedding_function is None:
            print("BgeM3EmbedderAdapter: BGEM3EmbeddingFunction is not available or failed to initialize. Generate operations will use mock embeddings.")
        else:
             print("BgeM3EmbedderAdapter: Adapter successfully initialized with BGEM3EmbeddingFunction.")


    # Hugging Face transformers 사용 시 필요한 풀링 함수는 이제 필요 없습니다.

    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[EmbeddingVector]:
        print(f"[EMBEDDING] 시작: {len(chunks)}개 청크 임베딩")
        
        if not chunks:
             print("BgeM3EmbedderAdapter: No chunks to embed. Skipping embedding generation.")
             return [] # 청크가 없으면 빈 목록 반환

        # --- 임베딩 생성 로직 또는 폴백 로직 실행 ---

        # self._embedding_function 인스턴스가 유효하면 실제 임베딩 실행
        if self._embedding_function is not None:
            print("BgeM3EmbedderAdapter: Using configured BGEM3EmbeddingFunction.")
            try:
                # --- 1단계: DocumentChunk 목록에서 텍스트 내용 목록 준비 ---
                chunk_contents = [chunk.content for chunk in chunks]
                print(f"   Encoding {len(chunk_contents)} text snippets...")

                # --- 2단계: 실제 BGEM3EmbeddingFunction encode 메서드 호출 ★★★ ---
                # BGEM3EmbeddingFunction 클래스의 encode 메서드는 실제로 다른 이름일 수 있습니다.
                # 일반적으로 임베딩 모델은 'encode', 'encode_queries', 'get_embeddings' 등의 메서드를 사용합니다.

                # 다음과 같이 임베딩 메서드 호출 부분을 시도해 볼 수 있는 메서드들을 테스트합니다:
                try:
                    # 1. encode 메서드 시도
                    if hasattr(self._embedding_function, 'encode'):
                        embedding_results = self._embedding_function.encode(chunk_contents)
                    # 2. embed 메서드 시도
                    elif hasattr(self._embedding_function, 'embed'):
                        embedding_results = self._embedding_function.embed(chunk_contents)
                    # 3. get_embeddings 메서드 시도
                    elif hasattr(self._embedding_function, 'get_embeddings'):
                        embedding_results = self._embedding_function.get_embeddings(chunk_contents)
                    # 4. encode_queries 메서드 시도 (일부 모델에서 사용)
                    elif hasattr(self._embedding_function, 'encode_queries'):
                        embedding_results = self._embedding_function.encode_queries(chunk_contents)
                    # 5. __call__ 메서드 시도 (callable 객체일 경우)
                    elif callable(self._embedding_function):
                        embedding_results = self._embedding_function(chunk_contents)
                    else:
                        # 사용 가능한 메서드가 없으면 오류 발생
                        available_methods = [method for method in dir(self._embedding_function) 
                                            if not method.startswith('_') and callable(getattr(self._embedding_function, method))]
                        error_msg = f"BGEM3EmbeddingFunction에 임베딩 생성 메서드를 찾을 수 없습니다. 사용 가능한 메서드: {available_methods}"
                        print(error_msg)
                        raise AttributeError(error_msg)
                except Exception as e:
                    print(f"BgeM3EmbedderAdapter: 임베딩 생성 중 오류 발생 - {e}")
                    raise EmbeddingError(f"BGEM3EmbeddingFunction으로 임베딩 생성 실패: {e}") from e

                # --- 호출 결과는 임베딩 벡터 리스트 (List[List[float]]) 입니다. ---
                print(f"   Successfully generated {len(embedding_results)} raw vectors using BGEM3EmbeddingFunction.")

                # 생성된 벡터 수와 청크 수가 일치하는지 확인 (불일치 시 오류 가능성)
                if len(chunks) != len(embedding_results):
                    print(f"Warning: Number of chunks ({len(chunks)}) and generated embeddings ({len(embedding_results)}) do not match.")
                    # 이 경우 VectorDatabasePort.save_document_data에서 오류가 날 가능성이 높습니다.
                    # raise EmbeddingError("Chunk and embedding count mismatch after encoding.")


            except Exception as e: # BGEM3EmbeddingFunction 호출 중 발생할 수 있는 예외 처리
                 print(f"BgeM3EmbedderAdapter: Error during actual embedding generation - {e}")
                 # 임베딩 생성 중 발생한 예외를 어댑터 레벨의 EmbeddingError로 변환하여 다시 발생
                 raise EmbeddingError(f"Failed to generate embeddings using BGEM3EmbeddingFunction: {e}") from e


        else: # self._embedding_function 인스턴스가 없거나 초기화 실패 시 폴백 로직 실행
            print("BgeM3EmbedderAdapter: Using mock embedding generation (BGEM3EmbeddingFunction not available).")
            # ---> 모킹 벡터 생성 (이전과 동일) <---
            mock_dimension = 1024 # BGE-M3 차원 (1024로 추정)
            embedding_results = [[random.random() for _ in range(mock_dimension)] for _ in chunks]
            print(f"   Generated {len(embedding_results)} mock vectors.")


        # --- ★★★ 생성된 벡터 목록을 EmbeddingVector 도메인 모델 목록으로 변환 ★★★ ---
        # 어댑터의 책임: 외부 기술 결과(벡터 리스트)를 내부 도메인 모델(EmbeddingVector 리스트)로 매핑
        embeddings: List[EmbeddingVector] = []
        try:
            print("   Mapping vectors to EmbeddingVector domain models...")
            # 생성된 벡터와 해당 청크 메타데이터를 조합하여 EmbeddingVector 객체 생성
            # 청크 목록과 임베딩 벡터 목록은 같은 순서라고 가정합니다.
            for i, vector in enumerate(embedding_results):
                chunk_metadata_ref = chunks[i].metadata.copy() # 원본 청크 메타데이터 복사
                # EmbeddingVector 객체 생성
                embeddings.append(EmbeddingVector(vector=vector, metadata=chunk_metadata_ref))

            print(f"BgeM3EmbedderAdapter: Mapping complete. Created {len(embeddings)} EmbeddingVector objects.")

        except Exception as e: # 매핑 중 오류 발생 시 EmbeddingError 발생
             print(f"BgeM3EmbedderAdapter: Error mapping vectors to EmbeddingVector domain models - {e}")
             raise EmbeddingError(f"Failed to map embedding results to domain models: {e}") from e


        print(f"[EMBEDDING] 성공: {len(embeddings)}개 임베딩 벡터 생성")
        if embeddings:
            print(f"  첫 임베딩 벡터 차원: {len(embeddings[0].vector)}")

        # 생성된 EmbeddingVector 객체 목록을 반환
        return embeddings