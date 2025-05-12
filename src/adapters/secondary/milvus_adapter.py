# src/adapters/secondary/milvus_adapter.py

# --- Milvus 클라이언트 라이브러리 임포트 ---
# 실제 사용하는 pymilvus 버전에 따라 임포트 구문이나 클래스/메서드 이름이 다를 수 있습니다.
# 제공된 코드 스니펫에 있던 임포트들을 최대한 반영합니다.
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Iterable, Set
import time
import os
import json

# 로거 임포트
from src.config import logger

try:
    # 성공 스크립트의 임포트 방식을 따릅니다.
    from pymilvus import ( # <-- pymilvus 최상위 패키지에서 임포트 시도
        MilvusClient,
        Collection,
        connections,
        FieldSchema,
        CollectionSchema,
        DataType,
        utility,
    )
    from pymilvus.exceptions import MilvusException # 예외는 보통 exceptions 모듈에 있습니다.

    # BGEM3EmbeddingFunction은 model.hybrid 모듈에 있습니다 (제공된 BGE-M3 스니펫 참고)
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction # <-- BGEM3EmbeddingFunction 임포트 추가

    _milvus_library_available = True
    logger.info("pymilvus library imported successfully.")
except ImportError:
    # pymilvus 라이브러리를 설치하지 않았거나 임포트할 수 없는 경우
    # 더미 클래스와 플래그를 정의합니다.
    _milvus_library_available = False
    logger.warning("Warning: pymilvus library not found. MilvusAdapter will not be functional.")
    # --- pymilvus가 없을 경우 더미 클래스 정의 (에러 방지 및 시뮬레이션) ---
    # 제공된 코드 스니펫의 클래스들을 더미로 정의합니다.
    logger.info("   Using dummy Milvus classes.")
    class MilvusClient:
        # 제공된 코드 기반 생성자 시그니처 반영
        def __init__(self, uri=None, host=None, port=None, token=None, user=None, password=None, **kwargs):
             logger.info(f"   (Simulating MilvusClient initialization - Library not available) uri={uri}, host={host}, port={port}, user={user}")
             self._is_connected = True # 시뮬레이션을 위해 항상 연결된 것으로 가정
             self.uri = uri # URI 저장
             self.host = host # host 저장
             self.port = port # port 저장
             self.token = token # 토큰 저장
             self.user = user # 유저 저장
             self.password = password # 비번 저장

        def is_connected(self): return self._is_connected
        # insert 메서드 시뮬레이션 (제공된 코드는 insert를 사용)
        def insert(self, collection_name, data, **kwargs):
             logger.info(f"   (Simulating MilvusClient.insert to '{collection_name}' - Library not available)")
             if not data: return None
             import random
             if random.random() > 0.9: raise Exception("Simulated Milvus insert failure") # 시뮬레이션 실패
             num_entities = len(data)
             logger.info(f"   (Simulating successful insert of {num_entities} entities)")
             # insert 결과 시뮬레이션 (UUID 대신 더미 ID 반환)
             sim_ids = [uuid.uuid4().int for _ in range(num_entities)] # INT64 ID 시뮬레이션
             return sim_ids

        # upsert 메서드 시뮬레이션 (우리가 원래 사용하려던 것)
        def upsert(self, collection_name, data, **kwargs):
             logger.info(f"   (Simulating MilvusClient.upsert to '{collection_name}' - Library not available)")
             if not data: return None
             import random
             if random.random() > 0.9: raise Exception("Simulated Milvus upsert failure")
             num_entities = len(data)
             logger.info(f"   (Simulating successful upsert of {num_entities} entities)")
             return {"insert_count": num_entities, "delete_count": 0, "upsert_count": num_entities} # upsert 결과 시뮬레이션

        # search 메서드 시뮬레이션
        def search(self, collection_name, data, **kwargs):
             logger.info(f"   (Simulating MilvusClient.search in '{collection_name}' - Library not available)")
             return [] # 빈 검색 결과 시뮬레이션

        # describe_collection 시뮬레이션
        def describe_collection(self, collection_name, **kwargs):
             logger.info(f"   (Simulating MilvusClient.describe_collection('{collection_name}'))")
             # 더미 컬렉션 정보 반환 (제공된 코드 기반)
             return {'collection_name': collection_name, 'collection_id': 12345, 'auto_id': True, 'description': 'Dummy Collection', 'properties': {'collection.metadata': {'embedding_model': {'name': 'dummy-model', 'dimension': 768}}}}

        @staticmethod # 스태틱 메서드 시뮬레이션
        def create_schema(auto_id=True, enable_dynamic_field=False, description="", **kwargs):
             logger.info("   (Simulating MilvusClient.create_schema)")
             class MockSchema:
                 def __init__(self, auto_id, enable_dynamic_field, description):
                      self.auto_id = auto_id
                      self.enable_dynamic_field = enable_dynamic_field
                      self.description = description
                      self.fields = []
                 def add_field(self, field_name, datatype, is_primary=False, auto_id=False, dim=None):
                      logger.info(f"      (Simulating schema.add_field: {field_name})")
                      self.fields.append({'name': field_name, 'datatype': str(datatype), 'is_primary': is_primary, 'auto_id': auto_id, 'dim': dim})
             # 더미 DataType 사용
             class MockDataType: INT64="INT64"; FLOAT_VECTOR="FLOAT_VECTOR"; JSON="JSON"; VARCHAR="VARCHAR"
             return MockSchema(auto_id, enable_dynamic_field, description)


    class Collection: # 더미 Collection 클래스
        def __init__(self, name, **kwargs): logger.info(f"   (Simulating Collection('{name}'))")
        def insert(self, data, **kwargs): logger.info("   (Simulating Collection.insert)"); return [] # 더미 결과
        def upsert(self, data, **kwargs): logger.info("   (Simulating Collection.upsert)"); return {"insert_count": 0} # 더미 결과
        def search(self, data, **kwargs): logger.info("   (Simulating Collection.search)"); return [] # 더미 결과
        def load(self): logger.info("   (Simulating Collection.load)")
        def release(self): logger.info("   (Simulating Collection.release)")
        def create_index(self, field_name, index_params, **kwargs): logger.info(f"   (Simulating Collection.create_index on {field_name})")
        def set_properties(self, properties): logger.info(f"   (Simulating Collection.set_properties)")
        def describe(self): return {'name': self.name} # 더미 describe

    class connections: # 더미 connections 클래스
        def connect(self, host, port, token, **kwargs): logger.info(f"   (Simulating connections.connect to {host}:{port})")
        def disconnect(self, alias="default"): logger.info("   (Simulating connections.disconnect)")
        def list_connections(self): logger.info("   (Simulating connections.list_connections)"); return ["default"]

    class utility: # 더미 utility 클래스
        def has_collection(self, collection_name, **kwargs): logger.info(f"   (Simulating utility.has_collection('{collection_name}'))"); return True # 항상 있다고 시뮬레이션
        def drop_collection(self, collection_name, **kwargs): logger.info(f"   (Simulating utility.drop_collection('{collection_name}'))")

    class FieldSchema: pass # 더미
    class CollectionSchema: pass # 더미
    class DataType: INT64="INT64"; FLOAT_VECTOR="FLOAT_VECTOR"; JSON="JSON"; VARCHAR="VARCHAR" # 더미 Enum 멤버

    class MilvusException(Exception): pass # 더미 예외


# --- 어댑터 특정 예외 정의 ---
class VectorDatabaseError(Exception):
    """벡터 데이터베이스와 상호작용 중 발생한 일반적인 오류를 나타냅니다."""
    pass

class MilvusAdapterError(VectorDatabaseError):
    """Milvus 어댑터와 관련된 특정 오류를 나타냅니다."""
    pass


import uuid # ID 생성을 위해 임포트
from ports.output_ports import VectorDatabasePort # 구현할 포트 임포트
from domain.models import DocumentChunk, EmbeddingVector # 저장할 도메인 모델 임포트
from typing import List, Dict, Any, Optional # 타입 힌트 임포트
import json


class MilvusAdapter(VectorDatabasePort):
    """
    Milvus 벡터 데이터베이스와 연동하여 VectorDatabasePort를 구현하는 어댑터.
    문서 청크와 임베딩 벡터를 Milvus 컬렉션에 저장합니다.
    """
    def __init__(self, host=None, port=None, collection_name="test_250430_1024_hybrid", token=None, uri=None):
        logger.info("MilvusAdapter: 수정된 방식으로 초기화 시작...")
        
        self._collection_name = collection_name
        self._is_initialized_successfully = False
        
        try:
            from pymilvus import MilvusClient
            
            # 명시적 URI 문자열 생성 - 앞에 tcp:// 스키마 사용
            milvus_uri = f"tcp://10.10.30.80:30953"
            logger.info(f"Milvus URI: {milvus_uri}")
            
            # MilvusClient 생성 시 uri를 첫 번째 위치 인수로 전달
            self._client = MilvusClient(
                uri=milvus_uri,  # 첫 번째 매개변수로 uri 전달
                user="root",
                password="smr0701!",
                secure=False     # SSL 비활성화
            )
            logger.info("MilvusClient 인스턴스 생성 시도")
            
            # 연결 상태 확인
            self._is_initialized_successfully = True
            logger.info("MilvusAdapter 초기화 완료")
            
            
        except Exception as e:
            logger.error(f"Milvus 초기화 오류: {e}")
            self._client = None
            self._is_initialized_successfully = False
            raise Exception(f"Milvus 연결 실패: {e}")


    # --- save_document_data 메서드 상세 구현 ---
    def save_document_data(self, chunks: List[DocumentChunk], embeddings: List[EmbeddingVector]) -> None:
        """
        문서 청크 목록과 해당하는 임베딩 벡터 목록을 Milvus 컬렉션에 저장합니다.
        """
        logger.info(f"[STORAGE] 시작: {len(chunks)}개 청크와 {len(embeddings)}개 임베딩 저장 시도")

        # 어댑터가 유효하고 초기화 성공했는지 확인
        if not self._is_initialized_successfully or self._client is None:
             # self._client.is_connected()는 매번 호출하기보다 초기화 상태와 클라이언트 존재 여부로 판단
            error_msg = "MilvusAdapter: Adapter not successfully initialized. Cannot save data."
            logger.error(error_msg)
            raise MilvusAdapterError(error_msg)


        # 청크와 임베딩 개수 일치 확인
        if len(chunks) != len(embeddings):
            error_msg = f"MilvusAdapter: Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)}). Must be 1:1."
            logger.error(error_msg)
            raise MilvusAdapterError(error_msg)

        # 저장할 데이터가 없는 경우 처리
        if not chunks:
            logger.info("MilvusAdapter: No data to save. Skipping Milvus operation.")
            return # 저장할 데이터가 없으면 바로 반환

        # --- 데이터 준비: DocumentChunk/EmbeddingVector -> Milvus 삽입 형식 ---
        # Milvus insert/upsert 메서드가 요구하는 데이터 형식으로 변환해야 합니다.
        # 제공된 코드 스니펫에서 사용된 컬렉션 스키마를 따릅니다.
        # ★★★ 제공된 코드 스니펫의 Milvus 컬렉션 스키마 (예시) ★★★
        # - id: INT64 (Primary Key, auto_id=True) - ★ID는 데이터에 포함시키지 않음★
        # - dense_embedding: FloatVector (Dimension=1024) - 임베딩 벡터
        # - sparse_embedding: JSON - sparse 임베딩 (현재 우리 모델에는 없음)
        # - metadata: JSON - 메타데이터 (딕셔너리 형태)

        entities_to_insert = []
        try:
            logger.info(f"   Preparing {len(chunks)} entities for Milvus insertion based on script schema...")
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # 각 엔티티에 필요한 필드 준비 (스키마 필드명 사용)

                # ID 필드는 Milvus가 auto_id로 자동 생성하므로 데이터에 포함시키지 않습니다.

                # dense_embedding 필드: 임베딩 벡터 데이터 (List[float])
                # BgeM3는 1024 차원이지만, 현재 EmbeddingVector는 List[float]만 가집니다. 차원 검증 필요시 추가.
                dense_vector_data = embedding.vector # 임베딩 벡터 데이터 (List[float])
                # TODO: 차원 검증 로직 추가 (예: if len(dense_vector_data) != 1024: raise ValueError(...))

                # sparse_embedding 필드: sparse 임베딩 (JSON)
                # 현재 EmbeddingVector 모델에는 sparse 데이터가 없습니다.
                # 제공된 코드 스니펫은 vectors라는 커스텀 타입에서 sparse를 가져옵니다.
                # 우리 모델에 sparse가 있다면 여기서 추출해야 합니다.
                # 예시: sparse_vector_data = embedding.sparse_vector # EmbeddingVector 모델에 sparse_vector 속성이 있다면
                sparse_vector_data = {} # 현재 모델에 없으므로 빈 딕셔너리로 시뮬레이션 (JSON 필드에 빈 객체 저장)
                # TODO: EmbeddingVector 모델에 sparse 임베딩 필드 추가 후 여기서 추출 로직 작성

                # metadata 필드: 메타데이터 (JSON)
                # DocumentChunk와 EmbeddingVector의 메타데이터에서 필요한 정보를 추출하여 JSON 객체(Dict)로 만듦
                # 제공된 코드 스니펫은 {"chunk_text": text, "source_pdf": pdf_filename} 형태로 메타데이터 JSON을 구성했습니다.
                # 우리는 더 많은 메타데이터를 가지고 있으므로 이를 포함합니다.
                chunk_metadata_dict = chunk.metadata.copy()
                chunk_metadata_dict.pop('__internal_docling_document__', None) # 내부 객체 제거

                embedding_metadata_dict = embedding.metadata.copy()
                # 필요시 임베딩 메타데이터에서 가져올 정보 추가 (예: 모델 이름)
                # embedding_model_info = embedding_metadata_dict.get('model_name')

                # Milvus metadata 필드에 저장할 최종 JSON 객체 구성
                milvus_metadata_value = {
                    "chunk_text": chunk.content, 
                    "source_file": chunk_metadata_dict.get('filename', 'unknown'),
                    "chunk_index": chunk_metadata_dict.get('chunk_index', i),
                    "page_number": chunk_metadata_dict.get('page_number'),
                    "original_metadata": chunk_metadata_dict,
                }

                # 엔티티 딕셔너리 생성 - CustomJSONEncoder 사용
                entity = {
                    "dense_embedding": dense_vector_data,
                    "sparse_embedding": sparse_vector_data,
                    "metadata": json.dumps(milvus_metadata_value, ensure_ascii=False, cls=CustomJSONEncoder),
                }
                entities_to_insert.append(entity)

            logger.info(f"[STORAGE] 준비된 엔티티: {len(entities_to_insert)}개")

        except Exception as e:
            error_msg = f"MilvusAdapter: Error preparing data for Milvus: {e}"
            logger.error(error_msg)
            # 데이터 준비 중 오류 발생 시 MilvusAdapterError 예외 발생
            raise MilvusAdapterError(error_msg) from e


        # --- 데이터 저장: Milvus 클라이언트의 insert 메서드 호출 ★ 실제 호출 ★ ---
        # 제공된 코드 스니펫은 client.insert()를 사용했습니다. auto_id=True 스키마와 함께 사용됩니다.
        # 사용하는 pymilvus 버전과 MilvusClient 객체 사용 방식에 따라 호출 코드가 다를 수 있습니다.

        try:
            # Milvus insert 호출 전
            logger.info(f"[STORAGE] Milvus insert 호출...")
            
            # insert 호출
            mutation_result = self._client.insert(
                collection_name=self._collection_name,
                data=entities_to_insert
            )
            
            # 성공 여부 로깅
            logger.info(f"[STORAGE] 성공: Milvus 응답 = {mutation_result}")

        except MilvusException as e: # Milvus 라이브러리 특정 예외 처리 (pymilvus.exceptions.MilvusException 등)
            error_msg = f"MilvusAdapter: Milvus operation error during insert: {e}"
            logger.error(error_msg)
            # Milvus 관련 오류 발생 시 어댑터 특정 예외를 발생시켜 유스케이스로 전달
            raise MilvusAdapterError(error_msg) from e
        except Exception as e: # 그 외 insert 호출 중 발생할 수 있는 예외 처리
            error_msg = f"MilvusAdapter: An unexpected error occurred during Milvus insert: {e}"
            logger.error(error_msg)
            # 예상치 못한 오류 발생 시 MilvusAdapterError 예외 발생
            raise MilvusAdapterError(error_msg) from e

        logger.info(f"MilvusAdapter: Save operation finished for {len(entities_to_insert)} entities.")

    # --- 검색 기능 메서드 (RAG 검색 단계 필요시 구현) ---
    # VectorDatabasePort에 search_similar_vectors 메서드가 있다면 여기서 구현합니다.
    # def search_similar_vectors(self, query_embedding: EmbeddingVector, top_k: int) -> List[DocumentChunk]:
    #    # ... Milvus 클라이언트의 search 메서드 호출 로직 구현 ...
    #    pass

# --- 더미 Milvus 클래스 정의 (pymilvus가 설치되지 않은 경우 사용) ---
# (앞서 정의된 더미 클래스들이 이 위치에 정의되어 있어야 합니다.)
# class MilvusClient: ...
# class Collection: ...
# class connections: ...
# class utility: ...
# class FieldSchema: ...
# class CollectionSchema: ...
# class DataType: ...
# class MilvusException(Exception): ...
# class VectorDatabaseError(Exception): ...
# class MilvusAdapterError(VectorDatabaseError): ...

# JSON 직렬화 가능한 커스텀 인코더 클래스 생성
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # TableData 및 기타 직렬화 불가능한 객체 처리
        try:
            return str(obj)
        except:
            return f"[Object of type {type(obj).__name__}]"