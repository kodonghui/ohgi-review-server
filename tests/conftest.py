"""
여러 테스트 파일이 함께 쓰는 준비물(fixture)

테스트 함수의 매개변수 이름을 fixture 함수 이름과 같게 적으면, pytest가 테스트마다 fixture를 새로 호출해서 넣어 준다.
그래서 한 테스트에서 리뷰를 등록해도 다음 테스트는 다시 리뷰 3개로 시작한다.

- repository      : 새 ReviewRepository. 미리 준비한 리뷰 3개로 시작한다.
- local_settings  : 대기 없는 가짜 모델과 테스트별 임시 저장 경로
- client          : 위 두 fixture로 FastAPI lifespan을 실행하는 TestClient

[실행 방법] 프로젝트 폴더에서
  uv run pytest -v          테스트 함수마다 결과를 한 줄씩 보여 준다
  uv run pytest --cov=app   app 폴더의 코드 중 테스트가 실행한 줄의 비율(커버리지)
"""

import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.core.config import Settings
from app.dependencies import get_review_repository
from app.main import app
from app.repositories.review_repository import ReviewRepository


@pytest.fixture
def repository() -> ReviewRepository:
    """테스트마다 새로 만드는 저장소. 미리 준비한 리뷰 3개로 시작한다."""

    return ReviewRepository()


@pytest.fixture
def local_settings(tmp_path, monkeypatch):
    """각 테스트의 매장 선호와 설정을 별도 임시 폴더에 둔다."""
    settings = Settings(
        _env_file=None, openai_api_key=None, fake_llm=True,
        fake_llm_delay_seconds=0, runtime_dir=tmp_path, thread_pool_size=None,
    )
    monkeypatch.setattr(app_main, "get_settings", lambda: settings)
    return settings


@pytest.fixture
def client(repository: ReviewRepository, local_settings, monkeypatch):
    monkeypatch.setitem(app.dependency_overrides, get_review_repository, lambda: repository)
    with TestClient(app) as http:
        yield http
