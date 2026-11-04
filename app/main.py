"""
오지 레스토랑 리뷰 서버 (FastAPI 미니 프로젝트 : 리뷰와 사장님 답글 API)

[실행 방법] 이 프로젝트 폴더에서
  uv run fastapi dev   ->  http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI

# FastAPI 앱 객체. uv run fastapi dev는 pyproject.toml의 entrypoint("app.main:app")를 보고 이 변수를 찾는다.
app = FastAPI(
    # /docs 화면 맨 위에 나오는 서버 이름과 버전
    title="오지 레스토랑 리뷰 서버",
    version="1.0.0",
)
