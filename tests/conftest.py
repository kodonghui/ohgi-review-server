"""
여러 테스트 파일이 함께 쓰는 준비물(fixture)

테스트 함수의 매개변수 이름을 fixture 함수 이름과 같게 적으면, pytest가 테스트마다 fixture를 새로 호출해서 넣어 준다.
그래서 한 테스트에서 리뷰를 등록해도 다음 테스트는 다시 리뷰 3개로 시작한다.

- repository      : 새 ReviewRepository. 미리 준비한 리뷰 3개로 시작한다.
- fake_llm_client : 기다리지 않고 바로 응답하는 가짜 LLM 클라이언트
- client          : 저장소와 LLM 클라이언트를 위 두 객체로 바꿔 끼운 TestClient

[실행 방법] 프로젝트 폴더에서
  uv run pytest -v          테스트 함수마다 결과를 한 줄씩 보여 준다
  uv run pytest --cov=app   app 폴더의 코드 중 테스트가 실행한 줄의 비율(커버리지)
"""

import pytest
from fastapi.testclient import TestClient

from app.clients.llm_client import FakeLlmClient
from app.core.config import Settings
from app.dependencies import get_llm_client, get_review_repository
from app.main import app
from app.repositories.review_repository import ReviewRepository


@pytest.fixture
def repository() -> ReviewRepository:
    """테스트마다 새로 만드는 저장소. 미리 준비한 리뷰 3개로 시작한다."""

    return ReviewRepository()


@pytest.fixture
def fake_llm_client() -> FakeLlmClient:
    """OpenAI를 호출하지 않고 바로 응답하는 가짜 LLM 클라이언트"""

    # Settings에 직접 전달한 값은 .env나 환경 변수보다 먼저 적용된다.
    return FakeLlmClient(Settings(fake_llm_delay_seconds=0))


@pytest.fixture
def client(repository: ReviewRepository, fake_llm_client: FakeLlmClient):
    """
    저장소와 LLM 클라이언트를 테스트용 객체로 바꿔 끼운 TestClient (Spring의 MockMvc)

    with 없이 만든 TestClient는 lifespan을 실행하지 않는다.
    그래서 OPENAI_API_KEY가 있는 PC에서 테스트해도 실제 OpenAI 클라이언트가 만들어지지 않는다.
    """

    # 1. 준비 : 의존성 함수를 테스트용 객체를 반환하는 함수로 바꾼다. (Spring의 @MockitoBean)
    app.dependency_overrides[get_review_repository] = lambda: repository
    app.dependency_overrides[get_llm_client] = lambda: fake_llm_client

    # 2. 테스트 실행
    yield TestClient(app)

    # 3. 정리 : 바꿔 끼운 의존성을 되돌려 다음 테스트에 영향을 남기지 않는다.
    app.dependency_overrides.clear()
