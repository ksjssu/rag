# RAG Hex System

헥사고날 아키텍처를 적용한 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 프로젝트 구조

이 프로젝트는 헥사고날 아키텍처를 따르며 다음과 같은 계층으로 구성되어 있습니다:

- Domain Layer: 핵심 비즈니스 로직 및 엔티티
- Application Layer: 유스케이스
- Ports Layer: 인터페이스 정의
- Adapters Layer: 외부 시스템과의 통합

## 설치 방법

```bash
pip install -r requirements.txt
```

## 환경 설정

1. `.env` 파일을 생성하고 필요한 API 키를 설정합니다.
2. `config.py`에서 기본 설정을 확인하고 필요한 경우 수정합니다.

## 실행 방법

```bash
uvicorn src.main:app --reload
``` 