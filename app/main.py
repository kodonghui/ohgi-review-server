"""
오지 레스토랑 리뷰 서버 (FastAPI 미니 프로젝트 : 리뷰와 사장님 답글 API)

손님 리뷰를 등록하고 조회하며, 저장된 리뷰를 LLM에 보내 사장님 답글 초안을 받는다.
- POST /api/reviews                         : 리뷰 등록 (def, 성공 201, 잘못된 값 422)
- GET  /api/reviews                         : 리뷰 전체 조회 (def)
- GET  /api/reviews/{review_id}             : 리뷰 하나 조회 (def, 없으면 404)
- POST /api/reviews/{review_id}/reply       : 사장님 답글 초안 생성 (async def + await, OpenAI 실패 502)
- POST /api/reviews/{review_id}/reply/sync  : 답글 초안 생성 동기 버전 (def, 비교용)

[실행 방법] 이 프로젝트 폴더에서
  uv run fastapi dev   ->  http://127.0.0.1:8000/docs
  OPENAI_API_KEY가 없거나 .env에 FAKE_LLM=true를 적으면 OpenAI 대신 가짜 LLM 클라이언트로 응답한다(비용 없음).
"""

from contextlib import asynccontextmanager

import anyio.to_thread
from fastapi import FastAPI

from app.clients.llm_client import FakeLlmClient, OpenAiLlmClient
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.routers import review_router


# [lifespan] 서버가 시작할 때(yield 이전)와 종료될 때(yield 이후) 실행할 코드를 한 함수에 적는다.
# (Spring의 @PostConstruct와 @PreDestroy를 한 함수에 정의한 것과 같다)
# 여기서 만든 LLM 클라이언트는 app.state에 두고 모든 요청이 함께 사용한다.
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # 1. 서버 시작 : 모든 요청이 함께 사용할 LLM 클라이언트를 만들어 app.state에 저장한다.
    # 1-1. FAKE_LLM=true거나 API 키가 없으면 OpenAI를 호출하지 않는 가짜 클라이언트를 만든다.
    if settings.fake_llm or settings.openai_api_key is None:
        print(f"[lifespan] 가짜 LLM 클라이언트를 사용한다 (FAKE_LLM={settings.fake_llm}, "
              f"키 있음={settings.openai_api_key is not None}, 응답까지 {settings.fake_llm_delay_seconds}초)")
        app.state.llm_client = FakeLlmClient(settings)
    # 1-2. 키가 있으면 실제 OpenAI 클라이언트를 만든다.
    else:
        print(f"[lifespan] OpenAI 클라이언트를 사용한다 (model={settings.openai_model})")
        app.state.llm_client = OpenAiLlmClient(settings)

    # 2. 실험용 : THREAD_POOL_SIZE가 있으면 def 핸들러를 실행하는 스레드 풀의 크기를 바꾼다.
    if settings.thread_pool_size is not None:
        anyio.to_thread.current_default_thread_limiter().total_tokens = settings.thread_pool_size
        print(f"[lifespan] 스레드 풀 크기 : {settings.thread_pool_size}개")

    # 3. 서버는 이 지점에서 멈춰 있는 동안 요청을 받는다.
    yield

    # 4. 서버 종료 : OpenAI 서버와의 연결을 닫는다. close()가 async def라서 await가 필요하다.
    await app.state.llm_client.close()


# FastAPI 앱 객체. uv run fastapi dev는 pyproject.toml의 entrypoint("app.main:app")를 보고 이 변수를 찾는다.
app = FastAPI(
    # /docs 화면 맨 위에 나오는 서버 이름과 버전
    title="오지 레스토랑 리뷰 서버",
    version="1.0.0",
    lifespan=lifespan,
)

# 예외 처리기를 등록한다. (Spring 컨테이너가 @ControllerAdvice를 알아서 관리하던 일을 직접 한다)
register_exception_handlers(app)

# 라우터를 등록한다. 등록해야 라우터에 적어 둔 주소가 실제로 생긴다.
app.include_router(review_router.router)
