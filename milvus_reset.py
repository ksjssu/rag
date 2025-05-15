from pymilvus import Collection, connections, utility
import time

def reset_milvus_collection():
    # Milvus 연결 정보
    host = "10.10.30.80"
    port = "30953"
    token = "root:smr0701!"
    collection_name = "test_250414"
    
    # Milvus 연결
    print(f"Milvus에 연결 중: {host}:{port}")
    connections.connect(host=host, port=port, token=token)
    
    # 컬렉션 존재 여부 확인 및 삭제
    if utility.has_collection(collection_name):
        print(f"컬렉션 '{collection_name}'이 존재합니다. 삭제 중...")
        utility.drop_collection(collection_name)
        print(f"컬렉션 '{collection_name}' 삭제 완료")
        time.sleep(2)  # 삭제 후 잠시 대기
    
    print(f"컬렉션이 성공적으로 삭제되었습니다. 이제 앱을 다시 시작하고 문서를 업로드하여 새로운 컬렉션을 생성하세요.")
    print("이 과정에서 올바른 인덱스가 자동으로 생성됩니다.")
    
    # 연결 종료
    connections.disconnect("default")
    print("Milvus 연결 종료")

if __name__ == "__main__":
    reset_milvus_collection() 