# src/adapters/secondary/milvus_adapter.py

import logging
import uuid
from typing import List, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# --- Milvus 클라이언트 라이브러리 임포트 ---
# 사용하는 pymilvus 버전에 따라 임포트 구문이나 클래스/메서드 이름이 다를 수 있습니다.
# 필요하다면 MilvusException 등 라이브러리 특정 예외도 여기서 임포트합니다.
try:
    # 예시: pymilvus >= 2.2.0 버전 기준 임포트
    from pymilvus import MilvusClient, Collection # Milvus 클라이언트 및 컬렉션 관련 클래스
    from pymilvus.exceptions import MilvusException # Milvus 라이브러리 예외 임포트 (오류 처리 시 사용)

    _milvus_library_available = True
    logger.info("pymilvus library imported.")
except ImportError:
    logger.warning("Warning: pymilvus library not found. MilvusAdapter will not be functional.")
    _milvus_library_available = False
    # --- pymilvus가 없을 경우 더미 클래스 정의 ---
    # (코드 하단에 실제 더미 클래스들이 정의되어 있어야 합니다. 여기서는 생략)
    logger.info("   Using dummy Milvus classes.")
    class MilvusClient: pass
    class Collection: pass
    class MilvusException(Exception): pass


# --- 어댑터 특정 예외 정의 ---
# VectorDatabasePort 정의 시 Raises 절에 명시된 예외입니다.
class VectorDatabaseError(Exception):
    """벡터 데이터베이스와 상호작용 중 발생한 일반적인 오류를 나타냅니다."""
    pass

class MilvusAdapterError(VectorDatabaseError):
    """Milvus 어댑터와 관련된 특정 오류를 나타냅니다."""
    pass


import uuid # ID 생성을 위해 임포트 (필요시)
from ports.output_ports import VectorDatabasePort # 구현할 포트 임포트
from domain.models import DocumentChunk, EmbeddingVector # 저장할 도메인 모델 임포트
from typing import List, Dict, Any, Optional # 타입 힌트 임포트


class MilvusAdapter(VectorDatabasePort):
    """
    Milvus 벡터 데이터베이스와 연동하여 VectorDatabasePort를 구현하는 어댑터.
    문서 청크와 임베딩 벡터를 Milvus 컬렉션에 저장합니다.
    """
    def __init__(
        self,
        host: str = "10.10.30.80",
        port: int = 30953,
        collection_name: str = "rag_chunks", # 데이터를 저장할 Milvus 컬렉션 이름
        user: Optional[str] = "root", # Milvus 사용자 계정 (인증 필요 시)
        password: Optional[str] = "smr0701!", # Milvus 비밀번호 (인증 필요 시)
        # --- Milvus 클라이언트 설정에 필요한 다른 파라미터 추가 ---
        # 사용하는 pymilvus 버전 문서를 확인하고, MilvusClient 생성자에 필요한
        # 추가 파라미터가 있다면 여기에 추가하세요. (예: uri, secure, client_timeout 등)
        # 예: uri: Optional[str] = None, secure: bool = False
    ):
        """
        MilvusAdapter 초기화 및 Milvus 연결 설정.

        Args:
            host: Milvus 서버 호스트 주소.
            port: Milvus 서버 포트.
            collection_name: 데이터를 저장할 Milvus 컬렉션 이름. 이 컬렉션은 미리 생성되어 있어야 합니다.
                           또는 어댑터 초기화 시 컬렉션 생성/확인 로직을 추가할 수 있습니다.
            user: Milvus 사용자 계정 (인증 필요 시).
            password: Milvus 비밀번호 (인증 필요 시).
            # 기타 pymilvus Client 생성자 파라미터 (Docstring 업데이트 필요)
        """
        logger.info(f"MilvusAdapter: Initializing for Milvus at {host}:{port}, collection '{collection_name}'...")

        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._user = user
        self._password = password
        # self._uri = uri # uri 파라미터를 받았다면 저장
        # self._secure = secure # secure 파라미터를 받았다면 저장

        self._client: Optional[MilvusClient] = None # Milvus 클라이언트 인스턴스
        self._is_initialized_successfully = False # 초기화 성공 여부 플래그

        if not _milvus_library_available:
            logger.warning("MilvusAdapter: pymilvus library not available. Adapter will use simulation.")
            # 라이브러리 없으면 더미 클라이언트 인스턴스 생성 (이전 더미 클래스 사용)
            # 더미 클래스 생성자 시그니처가 실제와 일치하도록 정의했는지 확인하세요.
            self._client = MilvusClient(host=self._host, port=self._port, user=self._user, password=self._password)
            if hasattr(self._client, 'is_connected') and self._client.is_connected():
                self._is_initialized_successfully = True
                logger.info("MilvusAdapter: Mock Milvus client initialized and connected.")
            else:
                 logger.error("MilvusAdapter: Mock client initialization failed or not connected.")
                 # 더미 초기화 실패 시 예외 발생 고려
                 # raise MilvusAdapterError("Failed to initialize mock Milvus client")


        else: # pymilvus 라이브러리 사용 가능 시 실제 연결 시도
            logger.info("MilvusAdapter: Attempting to connect to Milvus...")
            try:
                # --- ★★★ 실제 Milvus 클라이언트 연결 코드 ★★★ ---
                # 사용하는 pymilvus 버전에 따라 연결 방식이 다릅니다.
                # pymilvus 문서를 확인하여 정확한 MilvusClient 생성자 사용법을 따르세요.
                # host, port, user, password, uri, secure 등 파라미터 확인

                # 예시 (pymilvus >= 2.2.0, host/port 방식):
                self._client = MilvusClient(
                    uri=f"http://{self._host}:{self._port}",
                    user=self._user,
                    password=self._password,
                    # 필요하다면 secure=True, client_timeout 등 파라미터 추가 (pymilvus 문서 확인)
                )

                # 예시 (pymilvus < 2.2.0, uri 방식 또는 connect 메서드):
                # self._client = MilvusClient(uri=self._uri or f"tcp://{self._host}:{self._port}") # uri 사용 시
                # if self._user or self._password: # 인증 필요 시
                #    from pymilvus import Connection # Connection 클래스 임포트 필요
                #    Connection.connect(alias="default", host=self._host, port=self._port, user=self._user, password=self._password) # 연결 설정


                logger.info("MilvusAdapter: MilvusClient instance created.")

                # 연결 상태 확인 (필요시 is_connected() 메서드 사용)
                # is_connected() 메서드가 모든 pymilvus 버전에 있는지 확인하세요.
                if hasattr(self._client, 'is_connected') and self._client.is_connected():
                     logger.info("MilvusAdapter: Successfully connected to Milvus.")
                     self._is_initialized_successfully = True
                else:
                    logger.error("MilvusAdapter: Failed to connect to Milvus.")
                    # 연결 실패 시 MilvusAdapterError 예외 발생
                    raise MilvusAdapterError(f"Failed to connect to Milvus at {host}:{port}")


                # --- 컬렉션 확인 및 로드 로직 (필요시 __init__에 추가) ---
                # 데이터를 저장하기 전에 컬렉션이 존재하는지 확인하고 필요하면 생성하거나 로드해야 합니다.
                # 이 로직은 필수는 아니지만, 앱 시작 시 컬렉션 상태를 확인하는 데 유용합니다.
                try:
                     logger.info(f"Checking collection '{self._collection_name}'...")
                     # 실제 pymilvus 컬렉션 확인/생성/로드 로직 (pymilvus 문서 확인)
                     # 예시: client.has_collection, client.create_collection, client.load_collection
                     # 예: collection_exists = self._client.has_collection(collection_name=self._collection_name)
                     # 예: if not collection_exists: self._client.create_collection(collection_name=self._collection_name, schema=...) # 스키마 정의 필요
                     # 예: self._client.load_collection(collection_name=self._collection_name) # search/query 전에 로드 필요

                     # 여기서 실제 컬렉션 상태 확인 로직을 구현합니다.
                     # 예시: 컬렉션이 없으면 오류 발생
                     # if not self._client.has_collection(collection_name=self._collection_name):
                     #     raise MilvusAdapterError(f"Milvus collection '{self._collection_name}' does not exist.")

                     logger.info(f"Collection '{self._collection_name}' check/load finished (or will be handled on first save/query).")
                     # 컬렉션 확인/로드 성공 시에도 초기화 성공으로 간주
                     self._is_initialized_successfully = True # 최종 초기화 성공


                except Exception as e: # pymilvus 컬렉션 작업 중 발생할 수 있는 예외 처리
                     logger.warning(f"MilvusAdapter: Warning: Error during collection initialization for '{self._collection_name}': {e}")
                     # 컬렉션 관련 오류가 저장에 치명적이라면 여기서 예외 발생 고려
                     # self._client = None # 실패 시 클라이언트 None
                     # raise MilvusAdapterError(f"Error accessing collection '{self._collection_name}': {e}") from e
                     # 경고만 남기고 진행 시 어댑터 상태는 클라이언트 연결은 성공했으나 컬렉션 문제가 있을 수 있다고 알립니다.
                     # self._is_initialized_successfully 상태는 유지합니다.


            except Exception as e: # pymilvus 연결 자체 또는 초기 작업 중 발생할 수 있는 예외 처리
                logger.error(f"MilvusAdapter: Error during Milvus initialization: {e}")
                self._client = None # 초기화 실패 시 클라이언트 None
                self._is_initialized_successfully = False # 초기화 실패 플래그
                # 초기화 실패 시 MilvusAdapterError 예외 발생
                raise MilvusAdapterError(f"Error initializing Milvus adapter for {host}:{port}: {e}") from e


        if not self._is_initialized_successfully:
            logger.warning("MilvusAdapter: Milvus client is not available or failed to initialize successfully. Save operations will fail.")
        else:
             logger.info("MilvusAdapter: Adapter successfully initialized.")


    # --- save_document_data 메서드 상세 구현 ---
    def save_document_data(self, chunks: List[DocumentChunk], embeddings: List[EmbeddingVector]) -> None:
        """
        문서 청크 목록과 해당하는 임베딩 벡터 목록을 Milvus 컬렉션에 저장합니다.
        """
        logger.info(f"MilvusAdapter: Saving {len(chunks)} chunks and {len(embeddings)} embeddings to Milvus collection '{self._collection_name}'...")

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
        # 일반적으로 ID, 벡터, 메타데이터 필드를 포함하는 엔티티(Entity) 목록 형태입니다.
        # Milvus 컬렉션 스키마에 정의된 필드명과 타입에 맞춰 데이터를 준비해야 합니다.
        # ★★★ 가정된 Milvus 컬렉션 스키마 (예시) ★★★
        # - id: VarChar (Primary Key) - 우리의 DocumentChunk/EmbeddingVector 쌍을 식별
        # - vector: FloatVector (Dimension matching BGE-M3, e.g., 768) - 임베딩 벡터 자체
        # - text: VarChar (MAX_LENGTH 등 확인) - 청크 텍스트 내용 (검색 결과 반환 시 유용)
        # - source_file: VarChar - 원본 파일명 (원본 메타데이터에서 추출)
        # - chunk_index: Int64 - 원본 파일 내 청크 순서 (DocumentChunk 메타데이터에서 추출)
        # - page_number: Int64 (Optional) - 청크가 시작되는 페이지 (Docling 파싱 결과 메타데이터에서 추출)
        # - 其他 필요한 메타데이터 필드 (예: section, title 등) - Docling 메타데이터에서 추출하여 매핑
        # 스키마 필드명과 Python 데이터 타입 매핑 예시:
        # 'id': str (UUID)
        # 'vector': List[float]
        # 'text': str
        # 'source_file': str
        # 'chunk_index': int
        # 'page_number': Optional[int]

        entities_to_insert = []
        try:
            logger.info(f"Preparing {len(chunks)} entities for Milvus insertion...")
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # 각 엔티티에 필요한 필드 준비

                # ID 생성/가져오기: DocumentChunk/EmbeddingVector 쌍을 고유하게 식별할 ID
                # Milvus >= 2.2.0에서 VarChar PK를 사용한다면 UUID 문자열이 적합합니다.
                # DocumentChunk 메타데이터에 원본 문서 ID, 청크 ID 등이 있다면 활용할 수 있습니다.
                entity_id = str(uuid.uuid4()) # 예시 1: 어댑터에서 UUID 생성 (가장 흔함)
                # 예시 2: 청크 메타데이터에 'unique_chunk_id' 키가 있다면 사용, 없으면 UUID
                # entity_id = chunk.metadata.get('unique_chunk_id', str(uuid.uuid4()))
                # Milvus 자동 생성 PK 사용 시 ID 필드를 엔티티 딕셔너리에서 제외 (스키마 설정 확인)

                vector_data = embedding.vector # 임베딩 벡터 데이터 (List[float])

                # 메타데이터 필드 준비 및 매핑: DocumentChunk와 EmbeddingVector의 메타데이터에서 추출하여 스키마 필드에 매핑
                # ★★★ 사용하는 Milvus 컬렉션 스키마에 맞춰 아래 필드명과 데이터 추출/변환 로직을 정확히 수정해야 합니다. ★★★
                milvus_entity_data = {
                    "id": entity_id, # <-- 'id' 필드 (스키마에 맞게 필드명 수정)
                    "vector": vector_data, # <-- 'vector' 필드 (스키마에 맞게 필드명 수정)

                    # --- 메타데이터 필드 매핑 ---
                    "text": chunk.content, # <-- 'text' 필드에 청크 텍스트 저장 (VarChar)
                    "chunk_index": chunk.metadata.get('chunk_index', i), # <-- 'chunk_index' 필드에 저장 (Int64)
                    "source_file": chunk.metadata.get('filename', 'unknown'), # <-- 'source_file' 필드에 저장 (VarChar)
                    # 필요한 다른 메타데이터 필드 추가 (Docling 파싱 결과에서 얻은 page_number 등)
                    # Docling 청크 메타데이터(DocMeta)에서 추출한 정보 활용
                    # 예: 'page_number' 필드 (Int64, Optional) - 메타데이터에서 가져오고 타입 확인
                    "page_number": chunk.metadata.get('page_number') if isinstance(chunk.metadata.get('page_number'), int) else -1, # int가 아니면 -1 등으로 처리

                    # Docling meta (DocMeta 객체)의 headings, captions 등을 string 또는 JSON 형태로 저장할 수도 있습니다.
                    # 이 경우 해당 필드가 Milvus 스키마에 정의되어 있어야 합니다.
                    # "headings_text": str(chunk.metadata.get('headings')) if chunk.metadata.get('headings') is not None else "", # 예시 (VarChar)
                    # "docling_origin_info": str(chunk.metadata.get('origin')) if chunk.metadata.get('origin') is not None else "", # 예시 (VarChar)
                    # 임베딩 메타데이터에서 필요한 정보 (예: 모델명)
                    # "embedding_model_name": embedding.metadata.get('model_name') if embedding.metadata.get('model_name') is not None else "", # 예시 (VarChar)

                    # ★★★ 스키마에 맞지 않는 필드는 누락시키거나 정확한 타입으로 변환해야 합니다. ★★★
                    # VarChar의 경우 최대 길이 제한을 확인하고 필요시 텍스트를 잘라야 할 수 있습니다 (e.g., [:MilvusMaxCharLength]).
                    # bool, float, int, varchar, JSON 등의 필드 타입과 정확히 매핑해야 합니다.
                }

                # 엔티티 딕셔너리 생성 (Milvus insert/upsert 메서드가 요구하는 형식)
                # 필드명은 Milvus 스키마와 정확히 일치해야 합니다.
                # vector 필드명도 스키마와 일치해야 합니다.
                entity = {
                    "id": milvus_entity_data["id"],
                    "vector": milvus_entity_data["vector"], # 벡터 데이터는 리스트[float]
                    "text": milvus_entity_data["text"],
                    "chunk_index": milvus_entity_data["chunk_index"],
                    "source_file": milvus_entity_data["source_file"],
                    "page_number": milvus_entity_data["page_number"],
                    # 기타 스키마에 정의된 메타데이터 필드 추가
                    # "headings_text": milvus_entity_data["headings_text"],
                    # "docling_origin_info": milvus_entity_data["docling_origin_info"],
                    # "embedding_model_name": milvus_entity_data["embedding_model_name"],
                }
                entities_to_insert.append(entity)

            logger.info(f"Prepared {len(entities_to_insert)} entities for Milvus insertion.")

        except Exception as e:
            error_msg = f"MilvusAdapter: Error preparing data for Milvus: {e}"
            logger.error(error_msg)
            # 데이터 준비 중 오류 발생 시 MilvusAdapterError 예외 발생
            raise MilvusAdapterError(error_msg) from e


        # --- 데이터 저장: Milvus 클라이언트의 insert/upsert 메서드 호출 ★ 실제 호출 ★ ---
        # insert는 새 엔티티만 추가, upsert는 기존 ID가 있으면 업데이트
        # 여기서는 upsert를 사용하여 idempotent하게 저장하는 예시를 사용합니다.
        # upsert는 데이터 준비 형식(List[Dict])이 insert와 유사합니다.
        # 사용하는 pymilvus 버전과 MilvusClient 객체 사용 방식에 따라 호출 코드가 다를 수 있습니다.

        try:
            logger.info(f"Calling self._client.upsert() for collection '{self._collection_name}'...")
            # ★★★ 실제 Milvus 클라이언트 라이브러리 upsert 호출 라인 ★★★
            # self._client 객체를 통해 컬렉션의 upsert/insert 메서드를 호출합니다.
            # 사용하는 pymilvus 버전 문서를 반드시 확인하세요! (예: client.upsert, client.get_collection(...).insert)
            # 데이터 파라미터는 준비한 엔티티 목록(List[Dict])입니다.
            # 예시 (pymilvus >= 2.2.0):
            mutation_result = self._client.upsert( # <--- ▶︎▶︎▶︎ 실제 호출 라인! ◀︎◀︎◀︎
                 collection_name=self._collection_name, # 컬렉션 이름 전달
                 data=entities_to_insert, # <-- 준비한 엔티티 목록 전달 (List[Dict])
                 # 기타 upsert 메서드가 받는 파라미터 추가 (예: partition_name, consistency_level)
                 # pymilvus 문서 확인 필요
            )
            logger.info(f"MilvusAdapter: upsert operation completed. Result: {mutation_result}")

            # 저장 결과 확인
            if not mutation_result.insert_count and not mutation_result.upsert_count:
                logger.warning(f"MilvusAdapter: Upsert operation reported no inserts/updates: {mutation_result}")

        except MilvusException as e: # Milvus 라이브러리 특정 예외 처리 (pymilvus.exceptions.MilvusException 등)
            error_msg = f"MilvusAdapter: Milvus operation error during upsert: {e}"
            logger.error(error_msg)
            # Milvus 관련 오류 발생 시 어댑터 특정 예외를 발생시켜 유스케이스로 전달
            raise MilvusAdapterError(error_msg) from e
        except Exception as e: # 그 외 upsert 호출 중 발생할 수 있는 예외 처리
            error_msg = f"MilvusAdapter: An unexpected error occurred during Milvus upsert: {e}"
            logger.error(error_msg)
            # 예상치 못한 오류 발생 시 MilvusAdapterError 예외 발생
            raise MilvusAdapterError(error_msg) from e

        logger.info(f"MilvusAdapter: Save operation finished for {len(entities_to_insert)} entities.")

    # --- 검색 기능 메서드 (RAG 검색 단계 필요시 구현) ---
    # VectorDatabasePort에 search_similar_vectors 메서드가 있다면 여기서 구현합니다.
    # def search_similar_vectors(self, query_embedding: EmbeddingVector, top_k: int) -> List[DocumentChunk]:
    #    # ... Milvus 클라이언트의 search 메서드 호출 로직 구현 ...
    #    # 1. query_embedding.vector를 사용하여 Milvus search API 호출 (pymilvus 문서 확인)
    #    # 2. 검색 결과(Hit) 목록에서 엔티티 ID 및 스코어 가져옴
    #    # 3. 필요하다면 엔티티 ID를 사용하여 Milvus get API로 저장된 메타데이터(텍스트 포함) 조회
    #    # 4. 조회한 데이터로 DocumentChunk 객체 목록 생성
    #    # 5. DocumentChunk 목록 반환
    #    pass

# --- 더미 Milvus 클래스 정의 (pymilvus가 설치되지 않은 경우 사용) ---
# (앞서 정의된 더미 클래스들이 이 위치에 정의되어 있어야 합니다.)
# class MilvusClient: ...
# class Collection: ...
# class MilvusException(Exception): ...
# class VectorDatabaseError(Exception): ... # 어댑터 자체 예외는 위에서 정의됨
# class MilvusAdapterError(VectorDatabaseError): ... # 어댑터 특정 예외는 위에서 정의됨