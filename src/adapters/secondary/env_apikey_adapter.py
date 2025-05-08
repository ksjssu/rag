# src/adapters/secondary/env_apikey_adapter.py

import os
import logging
from ports.output_ports import ApiKeyManagementPort

# Configure logging
logger = logging.getLogger(__name__)

class EnvApiKeyAdapter(ApiKeyManagementPort):
    """
    환경 변수에서 API 키를 읽어와 ApiKeyManagementPort를 구현하는 어댑터.
    """
    def __init__(self):
        logger.info("EnvApiKeyAdapter initialized.")

    def get_api_key(self, service_name: str) -> str:
        """
        주어진 서비스 이름에 해당하는 환경 변수에서 API 키를 조회합니다.
        환경 변수 이름은 서비스 이름에 "_API_KEY"를 붙이는 규칙을 사용합니다.
        (예: "DO CLING" -> "DO CLING_API_KEY")
        """
        # 환경 변수 이름 규칙 정의 (대문자 변환, 공백/특수문자 처리 등)
        env_var_name = f"{service_name.upper().replace(' ', '_')}_API_KEY"

        api_key = os.getenv(env_var_name)

        if not api_key:
            # API 키가 없을 경우 예외 발생 (또는 다른 처리)
            raise ValueError(f"API Key for service '{service_name}' not found in environment variable '{env_var_name}'")

        logger.info(f"EnvApiKeyAdapter: Successfully retrieved API key for service '{service_name}'.")
        return api_key