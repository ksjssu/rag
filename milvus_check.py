from pymilvus import Collection, connections
import json

def check_milvus_data():
    # Milvus 연결 정보
    host = "10.10.30.80"
    port = "30953"
    token = "root:smr0701!"
    collection_name = "test_250414"
    
    # Milvus 연결
    print(f"Milvus에 연결 중: {host}:{port}")
    connections.connect(host=host, port=port, token=token)
    
    # 컬렉션 존재 여부 확인
    try:
        collection = Collection(collection_name)
        print(f"컬렉션 '{collection_name}' 연결 성공")
        
        # 컬렉션 로드 (데이터 조회 전 필수)
        collection.load()
        print(f"컬렉션 로드 완료")
        
        # 데이터 수 확인
        print(f"컬렉션 내 항목 수: {collection.num_entities}")
        
        # 샘플 데이터 가져오기
        if collection.num_entities > 0:
            print("\n데이터 샘플 조회 중...")
            results = collection.query(
                expr="document_id != ''", 
                output_fields=["document_id", "chunk_index", "content"],
                limit=3
            )
            
            if results:
                print(f"조회된 데이터 수: {len(results)}")
                for i, r in enumerate(results[:3], 1):
                    print(f"\n샘플 {i}:")
                    print(f"문서 ID: {r.get('document_id', 'N/A')}")
                    print(f"청크 인덱스: {r.get('chunk_index', 'N/A')}")
                    content = r.get('content', '')
                    print(f"내용 일부: {content[:100]}..." if content else "내용 없음")
            else:
                print("조회된 데이터 없음 (결과가 비어있음)")
        
        # 벡터 검색 테스트
        print("\n벡터 검색 테스트:")
        # 테스트용 임의 벡터(1024 차원)
        test_vector = [0.1] * 1024  
        search_results = collection.search(
            data=[test_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 10}},
            limit=2,
            output_fields=["document_id", "chunk_index", "content"]
        )
        
        if search_results and search_results[0]:
            print(f"검색 결과: {len(search_results[0])}개 항목")
            for i, hit in enumerate(search_results[0], 1):
                print(f"\n검색 결과 {i}:")
                print(f"문서 ID: {hit.entity.get('document_id', 'N/A')}")
                print(f"청크 인덱스: {hit.entity.get('chunk_index', 'N/A')}")
                print(f"유사도 점수: {hit.score:.4f}")
                content = hit.entity.get('content', '')
                print(f"내용 일부: {content[:100]}..." if content else "내용 없음")
        else:
            print("검색 결과 없음")
            
        # 컬렉션 정보 출력
        stats = collection.get_stats()
        print(f"\n컬렉션 통계: {stats}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
    
    # 연결 종료
    connections.disconnect("default")
    print("Milvus 연결 종료")

if __name__ == "__main__":
    check_milvus_data() 