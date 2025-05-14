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
    def __init__(self, host=None, port=None, collection_name="test_250414", token=None, uri=None):
        logger.info("MilvusAdapter: 초기화 시작...")
        
        self._collection_name = collection_name
        self._is_initialized_successfully = False
        
        try:
            from pymilvus import MilvusClient
            
            # uri가 제공되면 그대로 사용, 아니면 host와 port로 구성
            milvus_uri = uri
            if not milvus_uri and host and port:
                milvus_uri = f"tcp://{host}:{port}"
            
            logger.info(f"Milvus URI: {milvus_uri}")
            
            # 인증 정보 확인
            user = None 
            password = None
            if token and ':' in token:
                user, password = token.split(':', 1)
            
            # MilvusClient 생성
            self._client = MilvusClient(
                uri=milvus_uri,
                user=user,
                password=password,
                secure=False     # SSL 비활성화
            )
            logger.info("MilvusClient 인스턴스 생성 완료")
            
            # 연결 상태 확인
            self._is_initialized_successfully = True
            logger.info("MilvusAdapter 초기화 완료")
            
        except Exception as e:
            import traceback
            logger.error(f"Milvus 초기화 오류: {e}\n{traceback.format_exc()}")
            self._client = None
            self._is_initialized_successfully = False
            raise MilvusAdapterError(f"Milvus 연결 실패: {e}")

    def _create_collection(self, vector_dimension=1024):
        """
        지정된 이름과 스키마로 새로운 Milvus 컬렉션을 생성합니다.
        
        Args:
            vector_dimension: 임베딩 벡터의 차원(기본값 1024, BGE-M3 기준)
        """
        try:
            logger.info(f"[STORAGE] 컬렉션 {self._collection_name} 생성 시도 (차원: {vector_dimension})")
            
            # 스키마 생성
            schema = self._client.create_schema(
                auto_id=True,  # 자동 ID 생성
                enable_dynamic_field=True,  # 동적 필드 활성화
                description=f"RAG 애플리케이션을 위한 문서 컬렉션 (차원: {vector_dimension})"
            )
            
            # 필드 추가
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field(field_name="user_id", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=256)
            schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
            schema.add_field(field_name="group_list", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="metadata", datatype=DataType.JSON)
            schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="sparse_vector", datatype=DataType.VARCHAR, max_length=65535)  # 문자열로 저장
            schema.add_field(field_name="sparse_model", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)
            schema.add_field(field_name="dense_model", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="created_at", datatype=DataType.VARCHAR, max_length=32)
            schema.add_field(field_name="updated_at", datatype=DataType.VARCHAR, max_length=32)
            
            # 컬렉션 생성
            self._client.create_collection(
                collection_name=self._collection_name,
                schema=schema,
                consistency_level="Strong"
            )
            
            # 인덱스 생성 (dense_vector 필드에 대한 HNSW 인덱스)
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            }
            
            self._client.create_index(
                collection_name=self._collection_name,
                field_name="dense_vector",
                index_params=index_params
            )
            
            logger.info(f"[STORAGE] 컬렉션 {self._collection_name} 생성 완료")
            return True
        
        except Exception as e:
            logger.error(f"[STORAGE] 컬렉션 생성 실패: {e}")
            return False
    
    def _ensure_collection_exists(self, vector_dimension=1024):
        """
        컬렉션이 존재하는지 확인하고, 없으면 생성합니다.
        
        Args:
            vector_dimension: 임베딩 벡터의 차원
            
        Returns:
            bool: 컬렉션이 존재하거나 성공적으로 생성되었으면 True, 아니면 False
            dict: 컬렉션 스키마 정보 (필드 구조 등)
        """
        try:
            # 컬렉션 존재 여부 확인
            has_collection = False
            collection_info = None
            
            try:
                collection_info = self._client.describe_collection(collection_name=self._collection_name)
                if collection_info:
                    has_collection = True
                    logger.info(f"[STORAGE] 컬렉션 {self._collection_name} 이미 존재함")
            except Exception as e:
                logger.info(f"[STORAGE] 컬렉션 {self._collection_name} 존재하지 않음: {e}")
                has_collection = False
            
            # 컬렉션이 없으면 생성
            if not has_collection:
                logger.info(f"[STORAGE] 컬렉션 {self._collection_name} 생성 시작...")
                return self._create_collection(vector_dimension), None
            
            return True, collection_info
        
        except Exception as e:
            logger.error(f"[STORAGE] 컬렉션 존재 확인 중 오류: {e}")
            return False, None

    def _get_collection_fields(self, collection_info):
        """
        컬렉션 정보에서 필드 구조를 추출합니다.
        
        Args:
            collection_info: describe_collection 결과
            
        Returns:
            list: 컬렉션의 필드명 목록
        """
        try:
            if not collection_info:
                return []
                
            field_names = []
            
            # 컬렉션 정보에서 필드 이름 추출
            # schema 구조는 버전마다 다를 수 있으므로 여러 가능한 경로 시도
            if 'schema' in collection_info:
                schema = collection_info['schema']
                if 'fields' in schema:
                    field_names = [field.get('name') for field in schema['fields'] if 'name' in field]
            elif 'fields' in collection_info:
                field_names = [field.get('name') for field in collection_info['fields'] if 'name' in field]
                
            # 필드명을 찾지 못했을 경우 로그 출력
            if not field_names:
                logger.warning(f"[STORAGE] 컬렉션 필드명을 찾을 수 없음: {collection_info}")
                # 대체 방법으로 컬렉션에 있는 키들 그대로 반환
                return list(collection_info.keys())
                
            return field_names
            
        except Exception as e:
            logger.error(f"[STORAGE] 컬렉션 필드 추출 중 오류: {e}")
            return []

    # --- save_document_data 메서드 상세 구현 ---
    def save_document_data(self, chunks: List[DocumentChunk], embeddings: List[EmbeddingVector]) -> None:
        """
        문서 청크 목록과 해당하는 임베딩 벡터 목록을 Milvus 컬렉션에 저장합니다.
        """
        logger.info(f"[STORAGE] 시작: {len(chunks)}개 청크와 {len(embeddings)}개 임베딩 저장 시도")

        # 어댑터가 유효하고 초기화 성공했는지 확인
        if not self._is_initialized_successfully or self._client is None:
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
            
        # 임베딩 차원 확인 (첫 번째 임베딩 벡터 사용)
        vector_dimension = 1024  # 기본값
        if embeddings and hasattr(embeddings[0], 'vector') and len(embeddings[0].vector) > 0:
            vector_dimension = len(embeddings[0].vector)
            logger.info(f"[STORAGE] 임베딩 차원 감지: {vector_dimension}")
        
        # 컬렉션 존재 확인 및 생성
        exists, collection_info = self._ensure_collection_exists(vector_dimension)
        if not exists:
            error_msg = f"MilvusAdapter: Failed to ensure collection {self._collection_name} exists."
            logger.error(error_msg)
            raise MilvusAdapterError(error_msg)
            
        # 컬렉션 필드 구조 확인
        collection_fields = self._get_collection_fields(collection_info)
        logger.info(f"[STORAGE] 컬렉션 필드 구조: {collection_fields}")
        
        # 기존 컬렉션이 있는 경우 예상 필드 네이밍
        default_fields = ["dense_embedding", "sparse_embedding", "metadata"]
        
        # 필드 구조 분석 및 매핑 결정
        use_new_schema = False
        has_dynamic_field = False
        vector_field_name = "dense_vector"  # 기본값
        metadata_field_name = "metadata"    # 기본값
        
        # 컬렉션 정보에서 dynamic field 지원 여부 확인
        if collection_info and 'schema' in collection_info:
            schema = collection_info['schema']
            if 'enable_dynamic_field' in schema:
                has_dynamic_field = schema['enable_dynamic_field']
                logger.info(f"[STORAGE] 컬렉션 dynamic field 지원: {has_dynamic_field}")
                
        # 1. "dense_vector"가 있으면 새 스키마 사용
        if "dense_vector" in collection_fields:
            use_new_schema = True
            vector_field_name = "dense_vector"
        # 2. "dense_embedding"이 있으면 기존 스키마 사용
        elif "dense_embedding" in collection_fields:
            use_new_schema = False
            vector_field_name = "dense_embedding"
            
        logger.info(f"[STORAGE] 사용할 스키마: {'새 스키마' if use_new_schema else '기존 스키마'}, 벡터 필드: {vector_field_name}")

        # --- 데이터 준비: DocumentChunk/EmbeddingVector -> Milvus 삽입 형식 ---
        entities_to_insert = []
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            logger.info(f"   Preparing {len(chunks)} entities for Milvus insertion...")
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # 필수 메타데이터 추출
                chunk_metadata = chunk.metadata.copy()
                
                # 원본 문서 파일명 추출
                document_id = chunk_metadata.get('filename', 'unknown')
                
                # 청크 인덱스 추출
                chunk_index = chunk_metadata.get('chunk_index', i)
                
                # 컨텐츠 준비 (테이블 내용 포함)
                content = chunk.content

                # 처리 유형 분류 (텍스트/이미지/테이블)
                content_type = "text"  # 기본값
                if 'source' in chunk_metadata:
                    content_type = chunk_metadata['source']
                elif 'table_count' in chunk_metadata and chunk_metadata['table_count'] > 0:
                    content_type = "table"
                elif 'image_count' in chunk_metadata and chunk_metadata['image_count'] > 0:
                    content_type = "image"
                
                # 기존/새 스키마에 따라 데이터 준비
                if use_new_schema and (has_dynamic_field or "user_id" in collection_fields):
                    # 새로운 스키마: user_id, document_id, content 등 모든 필드 포함
                    # 간소화된 메타데이터 생성
                    simple_metadata = {
                        "source": document_id,
                        "index": chunk_index,
                        "type": content_type,
                        "page_number": chunk_metadata.get('page_number')
                    }
                    
                    # 스파스 벡터 처리 (문자열로 저장)
                    sparse_vector = "[]"
                    
                    # 엔티티 딕셔너리 생성
                    entity = {
                        "user_id": "user_001",
                        "document_id": document_id,
                        "chunk_index": chunk_index,
                        "group_list": "group_a",
                        "metadata": json.dumps(simple_metadata, ensure_ascii=False),
                        "content": content,
                        "sparse_vector": sparse_vector,
                        "sparse_model": "bge-m3",
                        vector_field_name: embedding.vector,
                        "dense_model": "bge-m3",
                        "created_at": current_time,
                        "updated_at": current_time
                    }
                else:
                    # 기존 스키마: dense_embedding, sparse_embedding, metadata만 포함
                    # 메타데이터에 모든 정보 포함
                    metadata_value = {
                        "chunk_text": content,
                        "source_file": document_id,
                        "chunk_index": chunk_index,
                        "page_number": chunk_metadata.get('page_number'),
                        "content_type": content_type,
                        "user_id": "user_001",
                        "group_list": "group_a",
                        "created_at": current_time
                    }
                    
                    # 엔티티 딕셔너리 생성 (기존 스키마)
                    entity = {
                        vector_field_name: embedding.vector,
                        "sparse_embedding": {},  # 기존 스키마 호환성
                        metadata_field_name: json.dumps(metadata_value, ensure_ascii=False)
                    }
                
                entities_to_insert.append(entity)

            logger.info(f"[STORAGE] 준비된 엔티티: {len(entities_to_insert)}개")
            if len(entities_to_insert) > 0:
                logger.info(f"[STORAGE] 엔티티 샘플 필드: {list(entities_to_insert[0].keys())}")

        except Exception as e:
            error_msg = f"MilvusAdapter: Error preparing data for Milvus: {e}"
            logger.error(error_msg)
            raise MilvusAdapterError(error_msg) from e

        # --- 데이터 저장: Milvus 클라이언트의 insert 메서드 호출 ---
        try:
            logger.info(f"[STORAGE] Milvus insert 호출...")
            
            # insert 호출
            mutation_result = self._client.insert(
                collection_name=self._collection_name,
                data=entities_to_insert
            )
            
            # 성공 여부 로깅
            logger.info(f"[STORAGE] 성공: Milvus 응답 = {mutation_result}")

        except MilvusException as e:
            error_msg = f"MilvusAdapter: Milvus operation error during insert: {e}"
            logger.error(error_msg)
            raise MilvusAdapterError(error_msg) from e
        except Exception as e:
            error_msg = f"MilvusAdapter: An unexpected error occurred during Milvus insert: {e}"
            logger.error(error_msg)
            raise MilvusAdapterError(error_msg) from e

        logger.info(f"MilvusAdapter: Save operation finished for {len(entities_to_insert)} entities.")

    # --- 검색 기능 메서드 (RAG 검색 단계 필요시 구현) ---
    # VectorDatabasePort에 search_similar_vectors 메서드가 있다면 여기서 구현합니다.
    # def search_similar_vectors(self, query_embedding: EmbeddingVector, top_k: int) -> List[DocumentChunk]:
    #    # ... Milvus 클라이언트의 search 메서드 호출 로직 구현 ...
    #    pass

# JSON 직렬화 가능한 커스텀 인코더 클래스 생성
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # TableData 및 기타 직렬화 불가능한 객체 처리
        try:
            return str(obj)
        except:
            return f"[Object of type {type(obj).__name__}]"