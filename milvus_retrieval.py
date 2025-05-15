from pymilvus import MilvusClient
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# 환경변수 또는 config에서 값 불러오기 추천
host = "10.10.30.80"
port = 30953
collection_name = "korean_pdf_files_test"
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

# 쿼리 텍스트 준비 및 임베딩
query_text = "인구구조 변화"
embedder = BGEM3EmbeddingFunction(model_name="BAAI/bge-m3")
query_vector = embedder.encode([query_text])[0]

# 벡터 검색 실행
results = client.search(
    collection_name=collection_name,
    data=[query_vector],
    limit=5,  # top 5
    output_fields=["document_id", "content"]
)

for hit in results[0]:
    print("document_id:", hit["document_id"], "score:", hit["score"])