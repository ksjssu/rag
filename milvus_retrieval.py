from pymilvus import MilvusClient
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# 환경변수 또는 config에서 값 불러오기 추천
host = "10.10.30.80"
port = 30953
collection_name = "test_250515"
user = "root"  # Milvus 기본 계정 (환경에 따라 다를 수 있음)
password = "smr0701!"  # 실제 비밀번호로 변경
# token = "..."  # 필요시

# MilvusClient 인스턴스 생성
client = MilvusClient(
    uri=f"http://{host}:{port}",
    user=user,
    password=password,
    # 또는 host=host, port=port, token=token 등으로도 가능
)

# BGE M3 임베딩 함수 인스턴스화
bge_m3_ef = BGEM3EmbeddingFunction(
    model_name='BAAI/bge-m3',  # 모델 이름 지정
    device='cpu',              # 사용할 장치 지정
    use_fp16=False             # 장치가 CPU인 경우 False로 설정
)

# 쿼리 텍스트 준비
# 사용자 입력으로 쿼리 텍스트 받기
user_query = input("검색할 내용을 입력하세요: ")
query_text = [user_query]

# 쿼리 임베딩 생성 - 문서에서 설명한 encode_queries 메서드 사용
query_embeddings = bge_m3_ef.encode_queries(query_text)

# 문서에 따르면 임베딩은 딕셔너리로 반환됨. 딕셔너리에서 'dense' 키 값 사용
query_vector = query_embeddings["dense"][0]

# 벡터 검색 실행
results = client.search(
    collection_name=collection_name,
    data=[query_vector],
    limit=5,  # top 5
    output_fields=["document_id", "content"]
)

print(f"\n=== 검색 결과: '{user_query}' ===")
print(f"총 {len(results[0])}개의 결과를 찾았습니다.\n")

for i, hit in enumerate(results[0]):
    print(f"[{i+1}] 문서: {hit['document_id']}")
    print(f"    유사도(거리): {hit.distance:.2f}")
    
    # 내용이 너무 길면 앞부분만 출력
    content = hit["content"].strip()
    if len(content) > 200:
        content = content[:200] + "..."
    print(f"    내용: {content}")
    print("-" * 80)