from pymilvus import (
    connections,
    Collection,
    utility
)
import logging

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

def get_top_vectors(col: Collection, limit: int = 5):
    """
    컬렉션의 최상위 벡터를 조회합니다.
    """
    try:
        # 먼저 출력 필드를 가져옴
        fields = get_schema_info(col)
        output_fields = []
        
        # 텍스트 관련 필드와 메타데이터 관련 필드를 찾음
        for field in fields:
            field_name = field.split(" ")[0]
            if field_name not in ["dense_vector", "sparse_vector"] and "id" not in field_name.lower():
                output_fields.append(field_name)
        
        # 출력 필드가 없으면 기본 필드만 출력
        if not output_fields:
            output_fields = []
            
        print(f"검색에 사용할 출력 필드: {output_fields}")
            
        # 빈 벡터로 검색하여 랜덤 결과 얻기
        results = col.search(
            data=[[0.0] * 1024],  # 1024차원의 0 벡터
            anns_field="dense_vector",
            limit=limit,
            output_fields=output_fields,
            param={
                "metric_type": "IP",
                "params": {"nprobe": 10}
            }
        )
        return results[0]
    except Exception as e:
        logger.error(f"벡터 조회 실패: {e}")
        return []

def main():
    # Milvus 연결 설정
    host = "10.10.30.80"
    port = 30953
    collection_name = "korean_pdf_files_test_sparse"
    user = "root"
    password = "smr0701!"

    # Milvus 연결
    connections.connect(
        alias="default",
        host=host,
        port=port,
        user=user,
        password=password
    )

    try:
        # 컬렉션 존재 여부 확인
        if not utility.has_collection(collection_name):
            logger.error(f"컬렉션 '{collection_name}'이 존재하지 않습니다.")
            return

        # 컬렉션 가져오기
        col = Collection(collection_name)
        col.load()

        # 컬렉션 통계 정보 조회 
        print("\n=== 컬렉션 정보 ===")
        print(f"컬렉션 이름: {collection_name}")
        
        try:
            # stats 메소드 시도
            stats = col.get_statistics()
            print(f"총 엔티티 수: {stats.get('row_count', '알 수 없음')}")
        except AttributeError:
            try:
                # 대체 메소드 시도
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

        # 최상위 벡터 조회
        print("\n=== 최상위 벡터 조회 결과 ===")
        results = get_top_vectors(col, limit=5)
        for i, hit in enumerate(results, 1):
            print(f"\n[결과 {i}]")
            # 모든 필드 출력
            for key, value in hit.items():
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."
                print(f"{key}: {value}")

    except Exception as e:
        logger.error(f"오류 발생: {e}")
    finally:
        # 컬렉션 언로드 및 연결 해제
        try:
            col.release()
        except:
            pass
        connections.disconnect("default")

if __name__ == "__main__":
    main() 