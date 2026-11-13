"""오지 레스토랑의 리뷰 CRUD와 LangGraph 답글 API."""

from contextlib import AsyncExitStack, asynccontextmanager
import logging

import anyio.to_thread
import httpx
from chromadb.errors import ChromaError
from fastapi import FastAPI, Request
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.sqlite.aio import AsyncSqliteStore

from app.clients.llm_client import FakeLlmClient, OpenAiLlmClient
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.repositories.review_repository import ReviewRepository
from app.routers import review_router
from app.schemas.reply_draft import GroundedReplyDraft
from app.schemas.review import ReplyPreferences
from app.services.hybrid_retriever import HybridRetriever
from app.services.policy_index import load_articles, open_policy_index
from app.services.policy_search import PolicySearch, PolicySearchResult
from app.services.policy_topics import PolicyEvidence
from app.services.reply_workflow import build_reply_workflow


logger = logging.getLogger(__name__)


def open_model_http_clients(timeout: float) -> tuple[httpx.Client, httpx.AsyncClient]:
    return httpx.Client(timeout=timeout), httpx.AsyncClient(timeout=timeout)


# [lifespan] 서버가 시작할 때(yield 이전)와 종료될 때(yield 이후) 실행할 코드를 한 함수에 적는다.
# (Spring의 @PostConstruct와 @PreDestroy를 한 함수에 정의한 것과 같다)
# 여기서 만든 LLM 클라이언트는 app.state에 두고 모든 요청이 함께 사용한다.
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    async with AsyncExitStack() as resources:
        if settings.fake_llm or settings.openai_api_key is None:
            llm = FakeLlmClient(settings)
        else:
            timeout = settings.openai_timeout_seconds
            sync_http, async_http = open_model_http_clients(timeout)
            resources.enter_context(sync_http)
            await resources.enter_async_context(async_http)
            llm = OpenAiLlmClient(settings, sync_http, async_http)
        app.state.llm_client = llm
        app.state.review_repository = ReviewRepository()

        settings.runtime_dir.mkdir(parents=True, exist_ok=True)
        store = await resources.enter_async_context(
            AsyncSqliteStore.from_conn_string(str(settings.runtime_dir / "preferences.sqlite3"))
        )
        await store.setup()
        app.state.reply_store = store
        app.state.reply_timeout = settings.reply_timeout_seconds

        try:
            index = await anyio.to_thread.run_sync(open_policy_index, settings.runtime_dir)
            articles = load_articles() if index is not None else []
            hybrid = HybridRetriever(index, articles) if index is not None else None
        except (OSError, ValueError, KeyError, RuntimeError, ChromaError) as error:
            logger.warning("규정 인덱스 로드 실패 (%s). 인덱스를 다시 만드세요.", type(error).__name__)
            index = hybrid = None
            articles = []
        if index is None:
            logger.info("규정 인덱스 없음. 규정이 필요한 요청은 매장 확인 상태로 처리합니다.")
        search = PolicySearch(index, hybrid, articles)
        serializer = JsonPlusSerializer(allowed_msgpack_modules=[
            (cls.__module__, cls.__name__)
            for cls in (ReplyPreferences, GroundedReplyDraft, PolicySearchResult, PolicyEvidence)
        ])
        app.state.reply_workflow = build_reply_workflow(
            search.retrieve, checkpointer=InMemorySaver(serde=serializer),
        )

        # 실험용 동기 핸들러의 스레드 풀 크기를 앱 수명 동안 적용한다.
        if settings.thread_pool_size is not None:
            limiter = anyio.to_thread.current_default_thread_limiter()
            resources.callback(setattr, limiter, "total_tokens", limiter.total_tokens)
            limiter.total_tokens = settings.thread_pool_size

        print(f"[lifespan] 답글 모델 {llm.model}")
        yield


# FastAPI 앱 객체. uv run fastapi dev는 pyproject.toml의 entrypoint("app.main:app")를 보고 이 변수를 찾는다.
app = FastAPI(
    # /docs 화면 맨 위에 나오는 서버 이름과 버전
    title="오지 레스토랑 리뷰 서버",
    version="1.0.0",
    lifespan=lifespan,
)

# 예외 처리기를 등록한다. (Spring 컨테이너가 @ControllerAdvice를 알아서 관리하던 일을 직접 한다)
register_exception_handlers(app)


@app.get("/health", include_in_schema=False)
async def health(request: Request) -> dict[str, str]:
    """부하 측정 도구가 모델 호출 없이 현재 모델을 확인할 수 있게 한다."""
    return {"model": request.app.state.llm_client.model}


# 라우터를 등록한다. 등록해야 라우터에 적어 둔 주소가 실제로 생긴다.
app.include_router(review_router.router)
app.include_router(review_router.preferences_router)
