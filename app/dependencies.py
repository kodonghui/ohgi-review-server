"""
의존성 함수 (Spring의 빈 등록 + 생성자 주입 역할)

핸들러의 매개변수에 Depends(의존성 함수)를 적어 두면, 요청이 들어올 때마다 FastAPI가 의존성 함수를 호출해서 반환값을 넣어 준다.
- ReviewRepository : 서버 전체에 하나만 있어야 하므로 이 모듈에서 한 번 만들어 두고 같은 객체를 반환한다.
- LLM 클라이언트    : main.py의 lifespan이 서버가 시작할 때 만들어 app.state에 넣어 둔 객체를 꺼낸다.
- 서비스            : 요청마다 새로 만든다. 저장소와 클라이언트만 들고 있는 가벼운 객체라서 괜찮다.

[POST /api/reviews/1/reply 요청 하나가 들어왔을 때]
create_reply(service: ReplyServiceDep)
    -> get_reply_service(repository, llm_client)
        -> get_review_repository()  : 서버에 하나뿐인 ReviewRepository 객체
        -> get_llm_client(request)  : lifespan이 만든 LLM 클라이언트
"""

from typing import Annotated

from fastapi import Depends, Request

from app.clients.llm_client import LlmClient
from app.repositories.review_repository import ReviewRepository
from app.services.reply_service import ReplyService
from app.services.review_service import ReviewService

# 이 모듈이 처음 import 될 때 한 번만 실행된다. 파이썬은 한 번 import 한 모듈을 다시 실행하지 않으므로,
# 모든 요청이 같은 ReviewRepository 객체(같은 딕셔너리)를 사용한다. (Spring의 싱글톤 빈)
review_repository = ReviewRepository()


# 이 함수 안에서 ReviewRepository()를 새로 만들면 요청마다 딕셔너리가 새로 생겨서 등록한 리뷰가 사라진다.
def get_review_repository() -> ReviewRepository:
    return review_repository


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
    repository: Annotated[ReviewRepository, Depends(get_review_repository)],
    llm_client: Annotated[LlmClient, Depends(get_llm_client)],
) -> ReplyService:
    return ReplyService(repository, llm_client)


ReplyServiceDep = Annotated[ReplyService, Depends(get_reply_service)]
