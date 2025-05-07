# 빌드 스테이지
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim as builder

# 메타데이터 추가
LABEL maintainer="SurroDocs Team"
LABEL description="SurroDocs Storage Service"
LABEL version="1.0"

# 작업 디렉토리 설정
WORKDIR /app

# Python 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

# 의존성 파일만 먼저 복사
COPY pyproject.toml uv.lock ./

# 의존성 설치
RUN uv sync --no-install-package surrodocs_common

# 소스 코드 복사
COPY . .

# 실행 스크립트 복사 및 권한 설정
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# 프로덕션 스테이지
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 파일만 복사
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# 환경 변수 설정
ENV PROFILE=prod \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 헬스체크
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]