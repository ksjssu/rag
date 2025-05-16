# src/adapters/secondary/bge_m3_embedder_adapter.py

import logging
import random
from typing import List, Dict, Any, Optional
import math
import numpy as np

# 로거 임포트
from src.config import logger

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
    logger.info("BGEM3EmbeddingFunction from pymilvus imported successfully.")
except ImportError:
    # pymilvus 라이브러리를 설치하지 않았거나 임포트할 수 없는 경우
    # 더미 클래스와 플래그를 정의합니다.
    _embedding_library_available = False # BGEM3EmbeddingFunction 사용 불가

    # --- 임베딩 라이브러리가 없을 경우 에러 방지를 위한 더미 클래스 정의 ---
    logger.info("   Using dummy embedding classes.")
    # Dummy class for BGEM3EmbeddingFunction (minimal implementation)
    class BGEM3EmbeddingFunction: # <-- 더미 클래스 정의
        # 제공된 코드 기반 생성자 시그니처 반영
        def __init__(self, model_name='dummy', device=None, use_fp16=False, batch_size=16, use_fast=True):
             logger.info(f"   (Simulating Dummy BGEM3EmbeddingFunction initialization - Library not available) Model: {model_name}, Device: {device}, use_fp16={use_fp16}, use_fast={use_fast}")
             self._model_name = model_name
             self._device = device
             self._use_fp16 = use_fp16
             self._use_fast = use_fast
             self._batch_size = batch_size
             self._mock_dimension = 1024 # BGE-M3 차원 (1024로 추정)

        # encode 메서드 시뮬레이션
        def encode(self, sentences): # <-- encode 메서드는 list[str] 입력만 받음 (제공된 코드 기반)
             logger.info(f"   (Simulating Dummy BGEM3EmbeddingFunction.encode) Encoding {len(sentences)} sentences...")
             import random # 모킹용
             # 결과는 List[List[float]] 형태로 시뮬레이션
             mock_embeddings = [[random.random() for _ in range(self._mock_dimension)] for _ in sentences]
             logger.info(f"   (Simulating successful encode of {len(sentences)} embeddings)")
             return mock_embeddings
             
        # encode_documents 메서드 시뮬레이션 (bge_m3.py 참고) - 딕셔너리 반환
        def encode_documents(self, documents):
            logger.info(f"   (Simulating Dummy BGEM3EmbeddingFunction.encode_documents) Encoding {len(documents)} documents...")
            import random # 모킹용
            
            # 결과는 Dictionary 형태로 시뮬레이션 (dense와 sparse 모두 포함)
            mock_dense = [[random.random() for _ in range(self._mock_dimension)] for _ in documents]
            
            # sparse 벡터를 위한 mock lexical_weights 생성 (BGE-M3 방식과 유사하게)
            mock_lexical_weights = []
            sparse_dim = 30522  # 가상의 토크나이저 크기
            
            for _ in documents:
                # 각 문서마다 10-20개의 랜덤 토큰 인덱스 생성
                num_tokens = random.randint(10, 20)
                indices = random.sample(range(sparse_dim), num_tokens)
                # 각 인덱스에 랜덤 가중치 할당
                mock_dict = {str(idx): random.random() for idx in indices}
                mock_lexical_weights.append(mock_dict)
            
            mock_result = {
                "dense": mock_dense,
                "lexical_weights": mock_lexical_weights
            }
            
            logger.info(f"   (Simulating successful encode_documents with both dense and sparse vectors)")
            return mock_result

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
        logger.info(f"DummyEmbedderAdapter initialized for model '{model_name}' on device '{device}'.")
        self._model_name = model_name
        self._device = device
        self._mock_dimension = 1024 # BGE-M3 차원 (1024로 추정)

    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[EmbeddingVector]:
        logger.info(f"DummyEmbedderAdapter: Using mock embeddings for {len(chunks)} chunks...")
        if not chunks:
            return []
        # Mock embedding generation logic (same as before)
        mock_embeddings_list = [[random.random() for _ in range(self._mock_dimension)] for _ in chunks]
        
        # sparse 벡터 목업 생성 (sparse_vector 필드에 Dict[int, float] 형태로)
        mock_sparse_vectors = []
        mock_colbert_vectors = []
        
        for _ in chunks:
            # 각 문서마다 5-10개의 랜덤 토큰 인덱스 생성
            num_tokens = random.randint(5, 10)
            sparse_dim = 30522  # 가상의 토크나이저 크기
            indices = random.sample(range(sparse_dim), num_tokens)
            # 각 인덱스에 랜덤 가중치 할당 (str 키가 아닌 int 키 사용)
            mock_sparse = {idx: random.random() for idx in indices}
            mock_sparse_vectors.append(mock_sparse)
            
            # 모의 colbert 벡터 생성 (토큰당 하나의 벡터)
            colbert_dim = 128  # ColBERT 임베딩 차원
            mock_colbert = [[random.random() for _ in range(colbert_dim)] for _ in range(random.randint(10, 20))]
            mock_colbert_vectors.append(mock_colbert)
        
        embeddings: List[EmbeddingVector] = []
        for i, vector in enumerate(mock_embeddings_list):
            chunk_metadata_ref = chunks[i].metadata.copy()
            embeddings.append(EmbeddingVector(
                vector=vector, 
                metadata=chunk_metadata_ref,
                sparse_vector=mock_sparse_vectors[i],
                colbert_vecs=mock_colbert_vectors[i]
            ))
        logger.info(f"DummyEmbedderAdapter: Generated {len(embeddings)} mock embeddings with sparse vectors and colbert vectors.")
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
        return_sparse: bool = True, # 스파스 벡터 반환 여부
        #api_key_port: Optional[ApiKeyManagementPort] = None, # BGEM3EmbeddingFunction은 토큰을 직접 받지 않는 것으로 추정
        # 기타 BGEM3EmbeddingFunction 초기화에 필요한 파라미터 추가 (pymilvus 문서 확인)
    ):
        """
        BgeM3EmbedderAdapter 초기화 및 BGEM3EmbeddingFunction 인스턴스 생성.

        Args:
             model_name: BGEM3EmbeddingFunction에 전달될 모델 이름.
             device: BGEM3EmbeddingFunction에 전달될 장치 ('cpu', 'cuda', 'etc').
             use_fp16: BGEM3EmbeddingFunction에 전달될 FP16 사용 여부. CPU 사용 시 False 권장.
             return_sparse: 스파스 벡터 반환 여부 (True/False)
             # 기타 BGEM3EmbeddingFunction 초기화 파라미터들 (pymilvus 문서 확인 필요)
        """
        logger.info(f"BgeM3EmbedderAdapter: Initializing for model '{model_name}' on device '{device or 'auto'}' with use_fp16={use_fp16}...")

        self._model_name = model_name
        self._device = device
        self._use_fp16 = use_fp16
        self._return_sparse = return_sparse
        # self._api_key_port = api_key_port # API 키 포트는 여기서 직접 사용되지 않는 것으로 보임 (BGEM3EmbeddingFunction이 내부적으로 처리 추정)

        self._embedding_function: Optional[BGEM3EmbeddingFunction] = None

        # BGEM3EmbeddingFunction은 pymilvus 라이브러리에 포함되어 있으므로,
        # pymilvus 임포트 성공 여부(_milvus_library_available 플래그)를 확인하여 초기화합니다.
        # _milvus_library_available 플래그는 MilvusAdapter.py 파일에 정의되어 있습니다.
        # main.py에서 _milvus_library_available를 임포트하고 이 값을 사용하여
        # BgeM3EmbedderAdapter 또는 DummyEmbedderAdapter를 선택하도록 조립해야 합니다.
        # 이 어댑터 자체는 BGEM3EmbeddingFunction 클래스가 임포트 가능할 때만 초기화를 시도합니다.

        # --- ★★★ BGEM3EmbeddingFunction 인스턴스 생성 ★★★ ---
        # BGEM3EmbeddingFunction 클래스가 임포트 가능할 때만 이 코드가 실행됩니다.
        if 'BGEM3EmbeddingFunction' in globals() and isinstance(BGEM3EmbeddingFunction, type): # <-- 클래스가 정의되었는지 확인 (ImportError 안 났다는 의미)
             logger.info(f"   Attempting to instantiate BGEM3EmbeddingFunction for model '{self._model_name}'...")
             try:
                 # 제공된 코드 스니펫의 BGEM3EmbeddingFunction 생성자 시그니처 사용
                 self._embedding_function = BGEM3EmbeddingFunction(
                     model_name=self._model_name,
                     device=self._device,
                     use_fp16=self._use_fp16,
                     batch_size=16,  # 배치 사이즈 명시적 설정
                     return_dense=True,  # dense 벡터 항상 반환
                     return_sparse=self._return_sparse,  # sparse 벡터 반환 여부
                     return_colbert_vecs=True  # colbert 벡터 사용
                 )
                 logger.info(f"BgeM3EmbedderAdapter: BGEM3EmbeddingFunction instance created successfully with return_sparse={self._return_sparse}")
             except Exception as e: # BGEM3EmbeddingFunction 초기화 중 발생할 수 있는 예외 처리
                 logger.error(f"BgeM3EmbedderAdapter: Error initializing BGEM3EmbeddingFunction: {e}")
                 self._embedding_function = None # 초기화 실패 시 None
                 # 초기화 실패 시 EmbeddingError 예외 발생 고려
                 # raise EmbeddingError(f"Failed to initialize BGEM3EmbeddingFunction: {e}") from e
        else:
             logger.warning("BgeM3EmbedderAdapter: BGEM3EmbeddingFunction class not available. Likely pymilvus not fully imported.")
             self._embedding_function = None # 클래스 자체가 없으면 None


        if self._embedding_function is None:
            logger.info("BgeM3EmbedderAdapter: BGEM3EmbeddingFunction is not available or failed to initialize. Generate operations will use mock embeddings.")
        else:
             logger.info("BgeM3EmbedderAdapter: Adapter successfully initialized with BGEM3EmbeddingFunction.")


    # Hugging Face transformers 사용 시 필요한 풀링 함수는 이제 필요 없습니다.

    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[EmbeddingVector]:
        if not chunks:
            logger.info("BgeM3EmbedderAdapter: No chunks to embed. Skipping embedding generation.")
            return [] # 청크가 없으면 빈 목록 반환

        # 1단계: DocumentChunk 목록에서 텍스트 내용 목록 준비
        chunk_contents = [chunk.content for chunk in chunks]
        logger.info(f"   Preparing to encode {len(chunk_contents)} text snippets...")

        # 2단계: 임베딩 벡터 생성 (실제 모델 또는 모의 벡터)
        embedding_vectors = []
        sparse_vectors = []
        colbert_vectors = []

        # self._embedding_function 인스턴스가 유효하면 실제 임베딩 실행
        if self._embedding_function is not None:
            logger.info("BgeM3EmbedderAdapter: Using configured BGEM3EmbeddingFunction.")
            try:
                # 임베딩 생성 (encode_documents 메서드 사용)
                try:
                    # 먼저 encode_documents 메서드 시도
                    embedding_results_dict = self._embedding_function.encode_documents(chunk_contents)
                    # 딕셔너리에서 dense 임베딩 추출
                    embedding_vectors = embedding_results_dict["dense"]
                    logger.info(f"   Successfully generated dense vectors using encode_documents method.")
                    
                    # 로그 추가: 반환된 임베딩 결과 딕셔너리의 키 목록 출력
                    logger.info(f"   embed_documents returned keys: {list(embedding_results_dict.keys())}")
                    
                    # colbert_vecs 처리 (있는 경우)
                    if "colbert_vecs" in embedding_results_dict:
                        colbert_vectors = embedding_results_dict["colbert_vecs"]
                        logger.info(f"   Successfully extracted ColBERT vectors with shape: {colbert_vectors[0].shape if hasattr(colbert_vectors[0], 'shape') else 'Unknown'}")
                    else:
                        # colbert_vecs가 없는 경우 None으로 채움
                        colbert_vectors = [None] * len(embedding_vectors)
                        logger.info("   No colbert_vecs found in embedding results")
                    
                    # sparse 벡터 처리를 위한 부분: sparse 키 먼저 확인, 없으면 lexical_weights 확인
                    if self._return_sparse and "sparse" in embedding_results_dict:
                        try:
                            # sparse 키는 scipy.sparse.csr_array 형태로 반환
                            sparse_matrices = embedding_results_dict["sparse"]
                            logger.info(f"   Found sparse matrices in csr_array format")
                            
                            # sparse_matrices 타입 및 구조 정보 출력
                            logger.info(f"   Sparse matrices type: {type(sparse_matrices)}")
                            if hasattr(sparse_matrices, "shape"):
                                logger.info(f"   Sparse matrices shape: {sparse_matrices.shape}")
                            
                            # csr_array 형식을 Dict[int, float]로 변환
                            if hasattr(sparse_matrices, 'shape'):
                                # 단일 행렬인 경우
                                sparse_matrix = sparse_matrices
                                logger.info(f"   Combined sparse matrix shape: {sparse_matrix.shape}")
                                
                                # 각 행(문서)에 대한 Dict[int, float] 생성
                                for i in range(sparse_matrix.shape[0]):
                                    row_indices = sparse_matrix.indices[sparse_matrix.indptr[i]:sparse_matrix.indptr[i+1]]
                                    row_data = sparse_matrix.data[sparse_matrix.indptr[i]:sparse_matrix.indptr[i+1]]
                                    # 키(인덱스)-값(가중치) 매핑
                                    converted_dict = {int(idx): float(val) for idx, val in zip(row_indices, row_data)}
                                    sparse_vectors.append(converted_dict)
                            else:
                                # 여러 행렬들인 경우 (리스트 형태)
                                for i, sparse_matrix in enumerate(list(sparse_matrices)):
                                    # 리스트 내 각 행렬의 비영 원소 정보 추출
                                    row_indices = sparse_matrix.indices
                                    row_data = sparse_matrix.data
                                    # 키(인덱스)-값(가중치) 매핑
                                    converted_dict = {int(idx): float(val) for idx, val in zip(row_indices, row_data)}
                                    sparse_vectors.append(converted_dict)
                            
                            logger.info(f"   Successfully extracted {len(sparse_vectors)} sparse vectors from csr_array format.")
                            if sparse_vectors and len(sparse_vectors) > 0 and sparse_vectors[0]:
                                logger.info(f"   Sample sparse vector has {len(sparse_vectors[0])} non-zero elements")
                                
                        except Exception as sparse_extract_error:
                            logger.warning(f"   Failed to extract sparse vectors from csr_array: {sparse_extract_error}")
                            sparse_vectors = [None] * len(embedding_vectors)
                    
                    # lexical_weights 키에 있는 경우 (fallback)
                    elif self._return_sparse and "lexical_weights" in embedding_results_dict:
                        # lexical_weights는 List[Dict[str, float]] 형태
                        lexical_weights = embedding_results_dict["lexical_weights"]
                        
                        # lexical_weights 데이터 샘플 출력
                        if lexical_weights and len(lexical_weights) > 0:
                            sample_weights = str(lexical_weights[0])[:100] + "..." if len(str(lexical_weights[0])) > 100 else str(lexical_weights[0])
                            logger.info(f"   Sample lexical_weights data: {sample_weights}")
                        
                        # 각 문서에 대한 lexical_weights를 Dict[int, float] 형태로 변환
                        for sparse_dict in lexical_weights:
                            # str 키를 int로 변환
                            converted_dict = {int(k): v for k, v in sparse_dict.items()}
                            sparse_vectors.append(converted_dict)
                        
                        logger.info(f"   Successfully extracted {len(sparse_vectors)} sparse vectors from lexical_weights.")
                    
                    # 어떤 키도 없는 경우
                    else:
                        # sparse 벡터가 없는 경우 None으로 채움
                        sparse_vectors = [None] * len(embedding_vectors)
                        if self._return_sparse:
                            import traceback
                            logger.warning("   No sparse or lexical_weights field found in embedding results.\n" + traceback.format_stack()[-2])
                
                except (AttributeError, TypeError) as method_error:
                    # encode_documents가 없거나 예상치 못한 인자를 받는 경우
                    logger.warning(f"encode_documents 메서드를 사용할 수 없습니다: {method_error}. encode 메서드로 대체합니다.")
                    # 대신 encode 메서드 시도 (sparse 벡터는 지원 안 될 수 있음)
                    result = self._embedding_function.encode(chunk_contents)
                    
                    # 반환 유형이 딕셔너리인지 확인
                    if isinstance(result, dict) and "dense" in result:
                        embedding_vectors = result["dense"]
                        
                        # sparse 벡터 처리
                        if self._return_sparse and "lexical_weights" in result:
                            lexical_weights = result["lexical_weights"]
                            for sparse_dict in lexical_weights:
                                converted_dict = {int(k): v for k, v in sparse_dict.items()}
                                sparse_vectors.append(converted_dict)
                        else:
                            sparse_vectors = [None] * len(embedding_vectors)
                    else:
                        # 단순히 벡터 리스트만 반환한 경우
                        embedding_vectors = result
                        sparse_vectors = [None] * len(embedding_vectors)
                    
                    logger.info(f"   Successfully generated vectors using encode method instead.")
                
                logger.info(f"   Generated {len(embedding_vectors)} dense vectors and {len(sparse_vectors)} sparse vectors.")
                
                # 생성된 벡터 수와 청크 수가 일치하는지 확인
                if len(chunks) != len(embedding_vectors):
                    logger.warning(f"Warning: Number of chunks ({len(chunks)}) and generated embeddings ({len(embedding_vectors)}) do not match.")
                    raise EmbeddingError("Chunk and embedding count mismatch after encoding.")
                
            except Exception as e:
                logger.error(f"BgeM3EmbedderAdapter: 임베딩 생성 중 오류 발생 - {e}")
                raise EmbeddingError(f"BGEM3EmbeddingFunction으로 임베딩 생성 실패: {e}") from e

        else:
            # 모델이 없는 경우 모의 임베딩 생성
            mock_dimension = 1024  # BGE-M3 차원 (1024로 추정)
            embedding_vectors = [[random.random() for _ in range(mock_dimension)] for _ in chunks]
            
            # 모의 sparse 벡터 생성
            sparse_dim = 30522  # 가상의 토크나이저 크기
            sparse_vectors = []
            colbert_vectors = []
            
            for _ in range(len(chunks)):
                if self._return_sparse:
                    # 각 문서마다 5-15개의 랜덤 토큰 인덱스 생성
                    num_tokens = random.randint(5, 15)
                    indices = random.sample(range(sparse_dim), num_tokens)
                    # 각 인덱스에 랜덤 가중치 할당
                    mock_sparse = {idx: random.random() for idx in indices}
                    sparse_vectors.append(mock_sparse)
                else:
                    sparse_vectors.append(None)
                
                # 모의 colbert 벡터 생성 (토큰당 하나의 벡터)
                colbert_dim = 128  # ColBERT 임베딩 차원
                mock_colbert = [[random.random() for _ in range(colbert_dim)] for _ in range(random.randint(10, 20))]
                colbert_vectors.append(mock_colbert)
                    
            logger.info(f"   Generated {len(embedding_vectors)} mock dense vectors, {len(sparse_vectors)} sparse vectors, and {len(colbert_vectors)} colbert vectors.")

        # 3단계: 생성된 벡터를 EmbeddingVector 도메인 모델로 변환
        try:
            logger.info("   Mapping vectors to EmbeddingVector domain models...")
            result = []
            for i, (vector, sparse_vector) in enumerate(zip(embedding_vectors, sparse_vectors)):
                if i < len(chunks):
                    try:
                        # sparse_vector가 있는 경우만 인자에 포함
                        if sparse_vector is not None:
                            result.append(EmbeddingVector(
                                vector=vector,
                                metadata=chunks[i].metadata.copy(),
                                sparse_vector=sparse_vector,
                                colbert_vecs=colbert_vectors[i] if i < len(colbert_vectors) and colbert_vectors[i] is not None else None
                            ))
                        else:
                            # sparse_vector가 없는 경우는 제외
                            result.append(EmbeddingVector(
                                vector=vector,
                                metadata=chunks[i].metadata.copy(),
                                colbert_vecs=colbert_vectors[i] if i < len(colbert_vectors) and colbert_vectors[i] is not None else None
                            ))
                    except TypeError as e:
                        # sparse_vector 매개변수가 지원되지 않는 경우 fallback
                        logger.warning(f"EmbeddingVector creation with sparse_vector and colbert_vecs failed, using dense only: {e}")
                        result.append(EmbeddingVector(
                            vector=vector,
                            metadata=chunks[i].metadata.copy()
                        ))
            
            logger.info(f"BgeM3EmbedderAdapter: Created {len(result)} EmbeddingVector objects.")
            logger.info(f"[EMBEDDING] 성공: {len(result)}개 임베딩 벡터 생성")
            if result:
                logger.info(f"  첫 임베딩 벡터 차원: {len(result[0].vector)}")
                if hasattr(result[0], 'sparse_vector') and result[0].sparse_vector:
                    logger.info(f"  첫 스파스 벡터 비영(non-zero) 요소 수: {len(result[0].sparse_vector)}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error during embedding mapping a- {e}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg)