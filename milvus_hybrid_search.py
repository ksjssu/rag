from pymilvus import (
    connections,
    Collection,
    utility,
    AnnSearchRequest,
    WeightedRanker
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction
import logging
import argparse
import numpy as np
from transformers import AutoTokenizer
import traceback

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_schema_info(col: Collection):
    """
    컬렉션의 스키마 정보를 조회합니다.
    """
    try:
        schema = col.schema
        field_names = []
        for field in schema.fields:
            field_names.append(f"{field.name} ({field.dtype})")
        return field_names
    except Exception as e:
        logger.error(f"스키마 조회 실패: {e}")
        return []

def convert_sparse_to_dict(sparse_vector):
    # 이미 딕셔너리 형태라면 그대로 사용
    if isinstance(sparse_vector, dict):
        return {int(k): float(v) for k, v in sparse_vector.items()}
    
    # scipy.sparse.coo_array 등의 COO 특성 직접 활용
    if hasattr(sparse_vector, 'data') and hasattr(sparse_vector, 'col'):
        print("COO 속성 직접 사용 (data/col)")
        result = {}
        # 데이터가 있는지 확인
        if len(sparse_vector.data) > 0:
            # 비영(non-zero) 요소별로 처리
            for i in range(len(sparse_vector.data)):
                col = sparse_vector.col[i]
                data = sparse_vector.data[i]
                result[int(col)] = float(data)
        return result
    
    # 마지막 시도: todok() 메서드 사용 (try-except로 안전하게)
    if hasattr(sparse_vector, 'todok'):
        try:
            print("todok() 메서드 시도")
            dok = sparse_vector.todok()
            result = {}
            # 딕셔너리 형태로 변환
            for k, v in dok.items():
                # 키가 튜플인 경우 (행, 열)
                if isinstance(k, tuple):
                    result[int(k[1])] = float(v)  # 열 인덱스 사용
                else:
                    # 키가 단일 값인 경우
                    result[int(k)] = float(v)
            return result
        except Exception as e:
            print(f"todok() 변환 실패: {e}")
            traceback.print_exc()
    
    # 최후의 대안
    print("알 수 없는 sparse_vector 형식, 토큰화 기반 벡터 생성 시도")
    # 이 부분에서는 토큰화 기반 벡터 생성 또는 빈 벡터 반환
    return {0: 0.0}

def dense_search(col, query_dense_embedding, limit=20, output_fields=None):
    """
    Dense vector를 사용한 검색
    """
    if output_fields is None:
        output_fields = []
        
    search_params = {
        "metric_type": "L2", 
        "params": {"nprobe": 10}
    }
    
    try:
        results = col.search(
            [query_dense_embedding],
            anns_field="dense_vector",
            limit=limit,
            output_fields=output_fields,
            param=search_params,
        )[0]
        return results
    except Exception as e:
        logger.error(f"Dense 검색 실패: {e}")
        return []

def get_sparse_vector_for_search(query_text):
    """
    BGE-M3 토크나이저를 사용하여 쿼리의 스파스 벡터를 생성합니다.
    이 방식은 저장된 문서와 동일한 형식의 스파스 벡터를 생성하여 검색 성능을 향상시킵니다.
    """
    try:
        print(f"\n직접 생성 방식으로 '{query_text}'의 스파스 벡터 생성 중...")
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", token="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")
        inputs = tokenizer(query_text, return_tensors="pt")
        
        # input_ids를 기반으로 스파스 벡터 생성
        token_ids = inputs.input_ids[0].tolist()
        sparse_dict = {}
        
        # 토큰 빈도수 계산
        token_counts = {}
        for token_id in token_ids:
            if token_id not in token_counts:
                token_counts[token_id] = 0
            token_counts[token_id] += 1
        
        # 토큰별 가중치 계산 (빈도에 기반)
        for token_id, count in token_counts.items():
            # 특수 토큰(BOS, EOS, PAD, UNK 등) 제외
            if token_id >= 5:  # 일반적으로 특수 토큰은 작은 ID를 가짐
                # 가중치 계산: 빈도와 1.0 사이의 값
                weight = min(1.0, 0.3 + 0.2 * count)  # 0.3~1.0 범위 (빈도에 따라 증가)
                sparse_dict[int(token_id)] = float(weight)
        
        print(f"생성된 스파스 벡터: 키 개수 {len(sparse_dict)}, 일부 샘플: {dict(list(sparse_dict.items())[:5])}")
        return sparse_dict
    except Exception as e:
        print(f"스파스 벡터 생성 중 오류: {e}")
        traceback.print_exc()
        return {0: 1.0}  # 오류 시 기본 벡터 반환

def sparse_search(col, query_sparse_embedding, limit=20, output_fields=None, use_direct_vector=False):
    """
    Sparse vector를 사용한 검색
    
    Args:
        col: Milvus Collection 객체
        query_sparse_embedding: 검색에 사용할 sparse 벡터 (scipy sparse 또는 딕셔너리)
        limit: 검색 결과 수
        output_fields: 결과에 포함할 필드 목록
        use_direct_vector: 변환 없이 쿼리 sparse 벡터를 그대로 사용할지 여부
    """
    if output_fields is None:
        output_fields = []
    
    # 디버깅 정보
    print("\n=== Sparse 검색 디버깅 ===")
    print(f"입력 sparse_vector 타입: {type(query_sparse_embedding)}")
    if hasattr(query_sparse_embedding, 'shape'):
        print(f"Shape: {query_sparse_embedding.shape}")
    if hasattr(query_sparse_embedding, 'nnz'):
        print(f"Non-zero 요소 수: {query_sparse_embedding.nnz}")
    
    # sparse vector를 Milvus 형식({인덱스: 값})으로 변환
    if use_direct_vector and isinstance(query_sparse_embedding, dict):
        sparse_dict = query_sparse_embedding  # 이미 딕셔너리면 그대로 사용
        print("직접 생성된 스파스 벡터 사용")
    else:
        sparse_dict = convert_sparse_to_dict(query_sparse_embedding)
    
    print(f"변환된 sparse_dict: {sparse_dict}")
    print(f"키 개수: {len(sparse_dict.keys())}")
    print(f"첫 5개 키-값 쌍: {list(sparse_dict.items())[:5]}")
    
    search_params = {
        "metric_type": "IP",
        "params": {},
    }
    
    try:
        results = col.search(
            [sparse_dict],
            anns_field="sparse_vector",
            limit=limit,
            output_fields=output_fields,
            param=search_params,
        )[0]
        return results
    except Exception as e:
        logger.error(f"Sparse 검색 실패: {e}")
        traceback.print_exc()
        return []

def hybrid_search(
    col,
    query_dense_embedding,
    query_sparse_embedding,
    sparse_weight=0.7,
    dense_weight=1.0,
    limit=20,
    output_fields=None,
    use_direct_sparse=False
):
    """
    Dense와 Sparse vector를 모두 사용한 하이브리드 검색
    
    Args:
        use_direct_sparse: 변환 없이 쿼리 sparse 벡터를 그대로 사용할지 여부
    """
    if output_fields is None:
        output_fields = []
    
    try:
        # sparse vector를 Milvus 형식으로 변환
        print("\n=== 하이브리드 검색 디버깅 ===")
        
        if use_direct_sparse and isinstance(query_sparse_embedding, dict):
            sparse_dict = query_sparse_embedding  # 이미 딕셔너리면 그대로 사용
            print("직접 생성된 스파스 벡터 사용")
        else:
            sparse_dict = convert_sparse_to_dict(query_sparse_embedding)
            
        print(f"변환된 sparse_dict: {sparse_dict}")
        print(f"키 개수: {len(sparse_dict.keys())}")
        print(f"첫 5개 키-값 쌍: {list(sparse_dict.items())[:5]}")
        
        # Dense 검색 요청
        dense_search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        dense_req = AnnSearchRequest(
            [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
        )
        
        # Sparse 검색 요청
        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_req = AnnSearchRequest(
            [sparse_dict], "sparse_vector", sparse_search_params, limit=limit
        )
        
        # 가중치 재랭커
        rerank = WeightedRanker(sparse_weight, dense_weight)
        
        # 하이브리드 검색 실행
        results = col.hybrid_search(
            [sparse_req, dense_req], 
            rerank=rerank, 
            limit=limit, 
            output_fields=output_fields
        )[0]
        
        return results
    except Exception as e:
        logger.error(f"하이브리드 검색 실패: {e}")
        traceback.print_exc()
        return []

def format_search_results(results, limit=None):
    """
    검색 결과를 포맷팅합니다.
    """
    if limit:
        results = results[:limit]
        
    formatted_results = []
    for i, hit in enumerate(results, 1):
        result_dict = {"rank": i, "distance": hit.get("distance", 0)}
        
        # document_id를 먼저 처리하여 상단에 표시
        document_id = None
        if "document_id" in hit:
            document_id = hit.get("document_id")
        elif "entity" in hit and isinstance(hit["entity"], dict) and "document_id" in hit["entity"]:
            document_id = hit["entity"].get("document_id")
        elif "metadata" in hit:
            # 메타데이터에서 소스 추출 시도
            try:
                metadata = hit.get("metadata", "{}")
                if isinstance(metadata, str) and metadata.strip().startswith("{"):
                    import json
                    metadata_dict = json.loads(metadata)
                    if "source" in metadata_dict:
                        document_id = metadata_dict["source"]
            except:
                pass
                
        if document_id:
            result_dict["document_id"] = document_id
            
        # 나머지 필드 처리
        for key, value in hit.items():
            if key not in ["distance"] and key not in result_dict:
                if isinstance(value, str) and len(value) > 300:
                    value = value[:300] + "..."
                result_dict[key] = value
                
        formatted_results.append(result_dict)
    
    return formatted_results

def apply_bge_reranker(query, hybrid_results, top_k=3):
    """
    BGE 리랭커를 적용하여 하이브리드 검색 결과를 다시 랭킹합니다.
    """
    # 리랭커 초기화
    print("\n=== BGE 리랭커 적용 ===")
    bge_rf = BGERerankFunction(
        model_name="BAAI/bge-reranker-v2-m3",  # 모델 이름 지정. 기본값은 `BAAI/bge-reranker-v2-m3`.
        device="cpu"  # 사용할 장치 지정, 예: 'cpu' 또는 'cuda:0'
    )
    
    # 하이브리드 검색 결과에서 텍스트 추출
    documents = []
    for hit in hybrid_results:
        # content 필드가 있는 경우 사용
        if "content" in hit:
            documents.append(hit["content"])
        # entity.content 필드가 있는 경우 사용
        elif "entity" in hit and isinstance(hit["entity"], dict) and "content" in hit["entity"]:
            documents.append(hit["entity"]["content"])
        # 그 외의 경우 빈 텍스트 사용
        else:
            documents.append("")
    
    if not documents:
        print("리랭킹할 문서가 없습니다.")
        return []
    
    # BGE 리랭커 적용
    print(f"쿼리: '{query}'로 {len(documents)}개 문서 리랭킹 중...")
    results = bge_rf(
        query=query,
        documents=documents,
        top_k=top_k,
    )
    
    return results

def main():
    # 인자 파싱
    parser = argparse.ArgumentParser(description="Milvus 하이브리드 검색")
    parser.add_argument("--host", default="10.10.30.80", help="Milvus 서버 호스트")
    parser.add_argument("--port", type=int, default=30953, help="Milvus 서버 포트")
    parser.add_argument("--collection", default="test_250520_normal", help="검색할 컬렉션 이름")
    parser.add_argument("--user", default="root", help="Milvus 사용자 이름")
    parser.add_argument("--password", default="smr0701!", help="Milvus 사용자 비밀번호")
    parser.add_argument("--query", default="", help="검색 쿼리")
    parser.add_argument("--sparse-weight", type=float, default=0.7, help="스파스 벡터 가중치")
    parser.add_argument("--dense-weight", type=float, default=1.0, help="덴스 벡터 가중치")
    parser.add_argument("--limit", type=int, default=5, help="검색 결과 수")
    parser.add_argument("--rerank", action="store_true", help="BGE 리랭커 적용 여부")
    parser.add_argument("--rerank-top", type=int, default=3, help="리랭킹 결과 상위 개수")
    parser.add_argument("--use-custom-sparse", action="store_true", help="토큰화 기반 스파스 벡터 사용")
    
    args = parser.parse_args()
    
    # 검색어가 없으면 입력 받기
    query = args.query
    if not query:
        query = input("검색어를 입력하세요: ")
    
    # Milvus 연결
    connections.connect(
        alias="default",
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password
    )

    try:
        # 컬렉션 존재 여부 확인
        if not utility.has_collection(args.collection):
            logger.error(f"컬렉션 '{args.collection}'이 존재하지 않습니다.")
            return

        # 컬렉션 가져오기
        col = Collection(args.collection)
        col.load()

        # 컬렉션 정보 조회
        print("\n=== 컬렉션 정보 ===")
        print(f"컬렉션 이름: {args.collection}")
        
        try:
            stats = col.num_entities
            print(f"총 엔티티 수: {stats}")
        except:
            print("총 엔티티 수: 알 수 없음")
                
        # 컬렉션 스키마 조회
        print("\n=== 컬렉션 스키마 정보 ===")
        fields = get_schema_info(col)
        print("필드 목록:")
        for field in fields:
            print(f"- {field}")
        
        # 출력 필드 설정 - document_id 추가 확인
        output_fields = []
        has_document_id = False
        for field in fields:
            field_name = field.split(" ")[0]
            if field_name not in ["dense_vector", "sparse_vector"] and "id" not in field_name.lower():
                output_fields.append(field_name)
            if field_name == "document_id":
                has_document_id = True
                
        # document_id가 필드에 있고 출력 필드에 아직 없으면 추가
        if has_document_id and "document_id" not in output_fields:
            output_fields.append("document_id")
            
        # metadata 필드가 있는지 확인하고 없으면 추가
        if "metadata" not in output_fields:
            for field in fields:
                if "metadata" in field.split(" ")[0]:
                    output_fields.append("metadata")
                    break
        
        # content 필드 확인
        has_content = False
        for field in fields:
            if "content" in field.split(" ")[0]:
                has_content = True
                if "content" not in output_fields:
                    output_fields.append("content")
                break
                
        if not has_content:
            logger.warning("컨텐츠 필드가 없습니다. 리랭킹이 제대로 작동하지 않을 수 있습니다.")
        
        if not output_fields:
            logger.warning("텍스트 또는 메타데이터 필드를 찾을 수 없습니다. 출력 필드가 제한될 수 있습니다.")
        
        print(f"검색에 사용할 출력 필드: {output_fields}")

        # BGE-M3 임베딩 함수 초기화
        print("\n임베딩 모델 로딩 중...")
        ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        
        # 쿼리 임베딩 생성
        print(f"검색어: {query}")
        query_embeddings = ef([query])
        
        # 커스텀 스파스 벡터 생성 (선택 사항)
        custom_sparse_vector = None
        if args.use_custom_sparse:
            custom_sparse_vector = get_sparse_vector_for_search(query)
        
        # 디버깅 정보 출력
        print(f"\nDense vector 차원: {len(query_embeddings['dense'][0])}")
        sparse_info = query_embeddings['sparse'][0]
        if hasattr(sparse_info, 'nnz'):
            print(f"모델 Sparse vector 요소 수: {sparse_info.nnz}")
        if custom_sparse_vector:
            print(f"커스텀 Sparse vector 요소 수: {len(custom_sparse_vector)}")
        
        # 1. Dense 검색
        print("\n=== Dense Vector 검색 결과 ===")
        dense_results = dense_search(col, query_embeddings["dense"][0], limit=args.limit, output_fields=output_fields)
        formatted_dense = format_search_results(dense_results)
        for result in formatted_dense:
            print(f"\n[Dense 결과 {result['rank']}]")
            # document_id를 먼저 출력
            if "document_id" in result:
                print(f"문서 ID: {result['document_id']}")
            print(f"유사도 거리: {result.get('distance', 'N/A')}")
            
            # 나머지 필드 출력 (중요 필드 먼저)
            important_fields = ["content", "metadata", "chunk_index"]
            for field in important_fields:
                if field in result and field not in ["document_id", "distance", "rank"]:
                    print(f"{field}: {result[field]}")
            
            # 그 외 나머지 필드 출력
            for key, value in result.items():
                if key not in important_fields + ["document_id", "distance", "rank"]:
                    print(f"{key}: {value}")
        
        # 2. Sparse 검색
        print("\n=== Sparse Vector 검색 결과 ===")
        if args.use_custom_sparse and custom_sparse_vector:
            # 커스텀 스파스 벡터 사용
            sparse_results = sparse_search(
                col, 
                custom_sparse_vector, 
                limit=args.limit, 
                output_fields=output_fields,
                use_direct_vector=True
            )
        else:
            # 모델 생성 스파스 벡터 사용
            sparse_results = sparse_search(
                col, 
                query_embeddings["sparse"][0], 
                limit=args.limit, 
                output_fields=output_fields
            )
            
        formatted_sparse = format_search_results(sparse_results)
        for result in formatted_sparse:
            print(f"\n[Sparse 결과 {result['rank']}]")
            # document_id를 먼저 출력
            if "document_id" in result:
                print(f"문서 ID: {result['document_id']}")
            print(f"유사도 거리: {result.get('distance', 'N/A')}")
            
            # 나머지 필드 출력 (중요 필드 먼저)
            important_fields = ["content", "metadata", "chunk_index"]
            for field in important_fields:
                if field in result and field not in ["document_id", "distance", "rank"]:
                    print(f"{field}: {result[field]}")
            
            # 그 외 나머지 필드 출력
            for key, value in result.items():
                if key not in important_fields + ["document_id", "distance", "rank"]:
                    print(f"{key}: {value}")
        
        # 3. 하이브리드 검색
        print("\n=== 하이브리드 검색 결과 ===")
        if args.use_custom_sparse and custom_sparse_vector:
            # 커스텀 스파스 벡터 사용
            hybrid_results = hybrid_search(
                col,
                query_embeddings["dense"][0],
                custom_sparse_vector,  # 커스텀 스파스 벡터
                sparse_weight=args.sparse_weight,
                dense_weight=args.dense_weight,
                limit=args.limit,
                output_fields=output_fields,
                use_direct_sparse=True
            )
        else:
            # 모델 생성 스파스 벡터 사용
            hybrid_results = hybrid_search(
                col,
                query_embeddings["dense"][0],
                query_embeddings["sparse"][0],
                sparse_weight=args.sparse_weight,
                dense_weight=args.dense_weight,
                limit=args.limit,
                output_fields=output_fields
            )
        
        formatted_hybrid = format_search_results(hybrid_results)
        for result in formatted_hybrid:
            print(f"\n[하이브리드 결과 {result['rank']}]")
            # document_id를 먼저 출력
            if "document_id" in result:
                print(f"문서 ID: {result['document_id']}")
            print(f"유사도 거리: {result.get('distance', 'N/A')}")
            
            # 나머지 필드 출력 (중요 필드 먼저)
            important_fields = ["content", "metadata", "chunk_index"]
            for field in important_fields:
                if field in result and field not in ["document_id", "distance", "rank"]:
                    print(f"{field}: {result[field]}")
            
            # 그 외 나머지 필드 출력
            for key, value in result.items():
                if key not in important_fields + ["document_id", "distance", "rank"]:
                    print(f"{key}: {value}")
        
        # 4. (선택적) BGE 리랭커 적용
        if args.rerank or '--rerank' in parser._option_string_actions:
            print("\n=== BGE 리랭커 적용 결과 ===")
            rerank_results = apply_bge_reranker(query, hybrid_results, top_k=args.rerank_top)
            
            for result in rerank_results:
                print(f"\n인덱스: {result.index}")
                print(f"점수: {result.score:.6f}")
                print(f"텍스트: {result.text[:300]}{'...' if len(result.text) > 300 else ''}")

    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
    finally:
        # 컬렉션 언로드 및 연결 해제
        try:
            col.release()
        except:
            pass
        connections.disconnect("default")

if __name__ == "__main__":
    main() 