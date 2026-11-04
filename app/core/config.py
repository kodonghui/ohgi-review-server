"""
설정값 (Spring의 application.yaml + @ConfigurationProperties 역할)

chap02의 config.py를 가져와 출력 토큰 상한의 기본값만 바꿨다.
- 필드 이름을 대문자로 바꾼 환경 변수를 자동으로 읽는다. openai_api_key 필드는 환경 변수 OPENAI_API_KEY를 읽는다.
- 값을 찾는 순서 : 1) 환경 변수 2) 이 프로젝트 폴더의 .env 파일 3) 아래 기본값
- Pydantic 모델이라서 타입과 조건을 검사한다. .env에 잘못된 값을 적으면 서버가 시작하면서 바로 알려 준다.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# 프로젝트 폴더(.env 파일이 있는 곳). 이 파일이 app/core/config.py이므로 세 단계 위다.
# 서버를 어느 폴더에서 실행하든 같은 .env 파일을 찾는다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        # .env 파일의 위치와 인코딩
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        # 이 클래스에 없는 환경 변수는 무시한다
        extra="ignore",
    )

    # OpenAI API 키. SecretStr은 print나 로그에 '**********'로 가려져 찍힌다. 실제 값은 get_secret_value()로 꺼낸다.
    # 기본값이 None이라 키가 없어도 서버는 실행되고, 그때는 가짜 LLM 클라이언트를 사용한다.
    openai_api_key: SecretStr | None = None

    # 사용할 모델
    openai_model: str = "gpt-6-luna"

    # 추론 정도. 사용 가능한 값을 Literal로 정해 두어 .env의 오타를 서버가 시작할 때 잡는다.
    openai_reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "none"

    # 출력 토큰 상한. 상한을 넘으면 응답이 중간에 끊긴다.
    # 답글은 세 문장에서 다섯 문장이라 chap02의 300보다 넉넉한 500으로 잡았다.
    # 추론 정도를 올리면 추론에 쓴 토큰도 이 상한에 포함된다. 끊긴 답글은 LLM 클라이언트가 502로 알린다.
    openai_max_output_tokens: int = Field(default=500, gt=0)

    # OpenAI의 응답을 기다리는 최대 시간(초)
    openai_timeout_seconds: float = Field(default=30.0, gt=0)

    # 가짜 LLM : true면 키가 있어도 OpenAI를 호출하지 않는다. 비용 없이 시험하거나 성능을 잴 때 사용한다.
    fake_llm: bool = False

    # 가짜 LLM이 응답하기 전에 기다리는 시간(초). 실제 LLM의 응답 시간을 흉내 낸다.
    fake_llm_delay_seconds: float = Field(default=3.0, ge=0)

    # 실험용 : def 핸들러를 실행할 스레드의 수. None이면 기본값(40개)을 그대로 사용한다.
    thread_pool_size: int | None = Field(default=None, ge=1)


# Settings()를 만들 때마다 .env를 다시 읽지 않도록 처음 만든 객체를 기억해 두고 계속 반환한다.
@lru_cache
def get_settings() -> Settings:
    return Settings()

# 확인 : 프로젝트 폴더에서 아래를 실행하면 읽어 들인 설정이 찍힌다. 키는 별표로 가려진다.
#   uv run python -c "from app.core.config import get_settings; print(get_settings())"
