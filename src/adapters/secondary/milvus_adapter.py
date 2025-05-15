# src/adapters/secondary/milvus_adapter.py

# --- Milvus 클라이언트 라이브러리 임포트 ---
# 실제 사용하는 pymilvus 버전에 따라 임포트 구문이나 클래스/메서드 이름이 다를 수 있습니다.
# 제공된 코드 스니펫에 있던 임포트들을 최대한 반영합니다.
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Iterable, Set
import time
import os
import json
from time import sleep

# 로거 임포트
from src.config import logger

_milvus_library_available = False

try:
    # 성공 스크립트의 임포트 방식을 따릅니다.
    from pymilvus import (
        connections, 
        Collection, 
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
        MilvusClient,
        MilvusException
    )

    _milvus_library_available = True
    logger.info("pymilvus library imported successfully.")
except ImportError:
    # pymilvus 라이브러리를 설치하지 않았거나 임포트할 수 없는 경우
    import traceback
    logger.warning("Warning: pymilvus library not found. MilvusAdapter will not be functional.")
    logger.warning(traceback.format_exc())
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
        """
        MilvusAdapter 초기화 및 Milvus 클라이언트 인스턴스 생성.
        """
        logger.info("MilvusAdapter: 초기화 시작...")
        
        self._is_initialized_successfully = False
        self._host = host
        self._port = port
        self._token = token
        self._collection_name = collection_name
        self._client = None  # Milvus 클라이언트 인스턴스

        try:
            # Milvus 클라이언트 인스턴스 생성
            # Milvus 클라이언트 인스턴스 생성
            
            # uri가 제공되면 그대로 사용, 아니면 host와 port로 구성
            milvus_uri = uri
            if not milvus_uri and host and port:
                milvus_uri = f"tcp://{host}:{port}"
                
            logger.info(f"Milvus URI: {milvus_uri}")
            
            # 서버 연결
            connections.connect(host=host, port=port, token=token)
            
            self._is_initialized_successfully = True
            logger.info("MilvusAdapter 초기화 완료")
            
        except Exception as e:
            import traceback
            logger.error(f"MilvusAdapter 초기화 실패: {e}\n{traceback.format_exc()}")
            self._is_initialized_successfully = False

    def _create_collection(self, vector_dimension=1024):
        """
        Milvus 컬렉션을 생성합니다.
        """
        logger.info(f"[STORAGE] '{self._collection_name}' 컬렉션 생성 시작...")
        
        try:
            # 필드 정의
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="group_list", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="sparse_vector", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="sparse_model", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dimension),
                FieldSchema(name="dense_model", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="updated_at", dtype=DataType.VARCHAR, max_length=64),
            ]
            
            # 컬렉션 스키마 생성
            schema = CollectionSchema(fields=fields, description=f"임베딩 벡터 저장용 {self._collection_name} 컬렉션")
            
            # 컬렉션 생성
            collection = Collection(name=self._collection_name, schema=schema)
            
            # 인덱스 생성 (dense_vector 필드에 대한 IVF_FLAT 인덱스)
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            
            collection.create_index(
                field_name="dense_vector",
                index_params=index_params
            )
            
            # 인덱스 생성 대기
            while not collection.has_index():
                logger.info("인덱스 생성 대기 중...")
                sleep(1)  # 1초 대기
            
            logger.info(f"[STORAGE] '{self._collection_name}' 컬렉션 생성 완료")
            return True
            
        except Exception as e:
            import traceback
            logger.error(f"[STORAGE] 컬렉션 생성 실패: {e}\n{traceback.format_exc()}")
            return False
    
    def _ensure_collection_exists(self, vector_dimension=1024):
        """
        Milvus 컬렉션이 존재하는지 확인하고, 없으면 생성합니다.
        """
        try:
            # 컬렉션 존재 여부 확인
            exists = utility.has_collection(self._collection_name)
            
            if not exists:
                # 컬렉션이 없으면 생성
                logger.info(f"[STORAGE] 컬렉션 '{self._collection_name}'이 존재하지 않습니다. 새로 생성합니다...")
                success = self._create_collection(vector_dimension)
                if not success:
                    raise MilvusAdapterError(f"Failed to create collection {self._collection_name}")
            else:
                # 컬렉션이 이미 존재하는 경우
                logger.info(f"[STORAGE] 컬렉션 '{self._collection_name}'이 이미 존재합니다.")
                
                # 컬렉션 로드 - 검색을 위해 메모리에 로드
                collection = Collection(self._collection_name)
                
                # 인덱스가 없으면 생성
                if not collection.has_index():
                    logger.info("기존 컬렉션에 인덱스가 없습니다. 인덱스를 생성합니다...")
                    index_params = {
                        "metric_type": "L2",
                        "index_type": "IVF_FLAT",
                        "params": {"nlist": 128}
                    }
                    collection.create_index(field_name="dense_vector", index_params=index_params)
                    
                    # 인덱스 생성 대기
                    while not collection.has_index():
                        logger.info("인덱스 생성 대기 중...")
                        sleep(1)
                        
                    logger.info("인덱스 생성 완료")
                
                # 컬렉션 로드 (필요한 경우)
                try:
                    collection.load()
                    logger.info(f"[STORAGE] 컬렉션 '{self._collection_name}' 로드 완료")
                except Exception as e:
                    logger.warning(f"[STORAGE] 컬렉션 로드 시 경고: {e}")
        except Exception as e:
            import traceback
            logger.error(f"[STORAGE] 컬렉션 {self._collection_name} 확인/생성 중 오류: {e}\n{traceback.format_exc()}")
            raise MilvusAdapterError(f"Failed to ensure collection {self._collection_name} exists.") from e

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
        문서 청크와 임베딩 벡터 정보를 Milvus에 저장합니다.
        """
        if not self._is_initialized_successfully:
            error_msg = "MilvusAdapter: Adapter not successfully initialized. Cannot save data."
            logger.error(error_msg)
            raise MilvusAdapterError(error_msg)
            
        if not chunks or not embeddings:
            logger.warning("[STORAGE] 저장할 청크 또는 임베딩이 없습니다.")
            return
            
        if len(chunks) != len(embeddings):
            error_msg = f"[STORAGE] 청크 수({len(chunks)})와 임베딩 수({len(embeddings)})가 일치하지 않습니다."
            logger.error(error_msg)
            raise MilvusAdapterError(error_msg)
            
        # 컬렉션 존재 여부 확인
        try:
            # 벡터 차원 확인 (첫 번째 임베딩 벡터 사용)
            vector_dimension = 1024  # 기본값
            if embeddings and hasattr(embeddings[0], 'vector') and len(embeddings[0].vector) > 0:
                vector_dimension = len(embeddings[0].vector)
                logger.info(f"[STORAGE] 임베딩 차원 감지: {vector_dimension}")
                
            self._ensure_collection_exists(vector_dimension)
        except Exception as e:
            import traceback
            logger.error(f"[STORAGE] 컬렉션 확인 중 오류: {e}\n{traceback.format_exc()}")
            raise MilvusAdapterError(f"MilvusAdapter: Failed to ensure collection {self._collection_name} exists.") from e
            
        # 삽입할 레코드 준비
        records = []
        
        try:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # 메타데이터는 JSON으로 직렬화
                metadata_copy = chunk.metadata.copy() if chunk.metadata else {}
                
                # 청크 인덱스 추출 - 변경: 먼저 chunk_index 값을 가져오기
                chunk_index = metadata_copy.get('chunk_index', i)
                
                # 소스(타입) 추출 - 청크 유형 명시적 파악
                chunk_source = metadata_copy.get('source', 'text')  # 기본값은 'text'
                
                # 청크 소스 유형에 대한 로깅 추가
                logger.info(f"  청크 {chunk_index} 유형: {chunk_source}")
                
                # 간소화된 메타데이터 생성
                simplified_metadata = {
                    "source": metadata_copy.get('filename', os.path.basename(metadata_copy.get('document_id', 'unknown'))),
                    "index": chunk_index,
                    "type": chunk_source  # 확인된 소스 타입 사용
                }
                
                # 테이블 타입인 경우 추가 로깅
                if chunk_source == 'table':
                    logger.info(f"  테이블 청크 {chunk_index} 인식 - 메타데이터에 테이블 타입으로 설정")
                    # 테이블 구조 정보가 있으면 로깅
                    if 'table_structure' in metadata_copy:
                        structure_preview = str(metadata_copy['table_structure'])[:100]
                        if len(str(metadata_copy['table_structure'])) > 100:
                            structure_preview += "..."
                        logger.info(f"  테이블 구조 정보: {structure_preview}")
                    
                    # 테이블 해시 정보 로깅 (있는 경우)
                    if 'table_hash' in metadata_copy:
                        logger.info(f"  테이블 해시: {metadata_copy['table_hash'][:8]}...")
                
                # 메타데이터 JSON 변환
                metadata_str = json.dumps(simplified_metadata, cls=CustomJSONEncoder)
                
                # 문서 ID 추출
                document_id = metadata_copy.get('document_id', metadata_copy.get('filename', 'unknown_document'))
                
                # 현재 시간
                current_time = time.strftime('%Y-%m-%d %H:%M:%S')
                
                # dense 벡터 추출
                dense_vector = embedding.vector
                
                # Milvus에 삽입할 레코드 생성
                record = {
                    "user_id": metadata_copy.get('user_id', 'default_user'),
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "group_list": metadata_copy.get('group_list', ''),
                    "metadata": metadata_str,
                    "content": chunk.content,
                    "sparse_vector": json.dumps([0.0] * 100),  # 빈 sparse 벡터
                    "sparse_model": "none",
                    "dense_vector": dense_vector,
                    "dense_model": "bge-m3",
                    "created_at": current_time,
                    "updated_at": current_time,
                }
                
                records.append(record)
            
            # 컬렉션 로드
            collection = Collection(self._collection_name)
            
            # 데이터 삽입
            logger.info(f"[STORAGE] Milvus에 {len(records)}개 레코드 삽입 시도 중...")
            result = collection.insert(records)
            
            # 변경사항 즉시 기록
            collection.flush()
            
            logger.info(f"[STORAGE] 성공: Milvus 응답 = {result}")
            
        except Exception as e:
            error_msg = f"[STORAGE] 데이터 저장 실패: {e}"
            logger.error(error_msg)
            raise MilvusAdapterError(error_msg) from e

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