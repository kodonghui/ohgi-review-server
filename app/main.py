"""
오지 레스토랑 리뷰 서버 (FastAPI 미니 프로젝트 : 리뷰와 사장님 답글 API)

손님 리뷰를 등록하고 조회한다.
- POST /api/reviews                         : 리뷰 등록 (def, 성공 201, 잘못된 값 422)
- GET  /api/reviews                         : 리뷰 전체 조회 (def)
- GET  /api/reviews/{review_id}             : 리뷰 하나 조회 (def, 없으면 404)

[실행 방법] 이 프로젝트 폴더에서
  uv run fastapi dev   ->  http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI

from app.core.exceptions import register_exception_handlers
from app.routers import review_router

# FastAPI 앱 객체. uv run fastapi dev는 pyproject.toml의 entrypoint("app.main:app")를 보고 이 변수를 찾는다.
app = FastAPI(
    # /docs 화면 맨 위에 나오는 서버 이름과 버전
    title="오지 레스토랑 리뷰 서버",
    version="1.0.0",
)

# 예외 처리기를 등록한다. (Spring 컨테이너가 @ControllerAdvice를 알아서 관리하던 일을 직접 한다)
register_exception_handlers(app)

# 라우터를 등록한다. 등록해야 라우터에 적어 둔 주소가 실제로 생긴다.
app.include_router(review_router.router)
