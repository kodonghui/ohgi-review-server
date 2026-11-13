"""lifespan에서 만든 저장소·모델·그래프를 요청 서비스에 전달한다."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request

from app.clients.llm_client import LlmClient
from app.domain.review import Review
from app.repositories.review_repository import ReviewRepository
from app.services.reply_service import ReplyService
from app.services.review_service import ReviewService


def get_review_repository(request: Request) -> ReviewRepository:
    return request.app.state.review_repository


def get_review_service(repository: Annotated[ReviewRepository, Depends(get_review_repository)]) -> ReviewService:
    return ReviewService(repository)


# 핸들러마다 Annotated[...]를 반복해서 적지 않도록 이름을 붙여 둔다. 핸들러에서는 service: ReviewServiceDep 한 줄이면 된다.
ReviewServiceDep = Annotated[ReviewService, Depends(get_review_service)]


# LLM 클라이언트는 OpenAI 서버와의 연결을 가지고 있어서 서버가 꺼질 때 닫아야 한다.
# 그래서 만들고 닫는 일을 lifespan 한곳에서 하고, 여기서는 꺼내기만 한다.
def get_llm_client(request: Request) -> LlmClient:
    return request.app.state.llm_client


# 답글 서비스는 리뷰를 꺼낼 저장소와 답글을 만들 LLM 클라이언트가 둘 다 필요하다.
def get_reply_service(
    request: Request,
    repository: Annotated[ReviewRepository, Depends(get_review_repository)],
    llm_client: Annotated[LlmClient, Depends(get_llm_client)],
) -> ReplyService:
    state = request.app.state
    return ReplyService(repository, llm_client, state.reply_workflow, state.reply_store, state.reply_timeout)


ReplyServiceDep = Annotated[ReplyService, Depends(get_reply_service)]


async def reserve_stream_review(review_id: int, service: ReplyServiceDep) -> AsyncIterator[Review]:
    # SSE 본문을 시작하기 전에 404·409를 판단하고 스트림이 닫힐 때 실행권을 반환한다.
    with service.repository.reply_operation(review_id) as review:
        yield review


StreamReplyDep = Annotated[Review, Depends(reserve_stream_review)]
