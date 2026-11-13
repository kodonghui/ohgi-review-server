"""리뷰 CRUD와 JSON·SSE 답글 API."""

import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import aclosing

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.core.exceptions import AiRequestError
from app.dependencies import ReplyServiceDep, ReviewServiceDep, StreamReplyDep
from app.schemas.review import (
    ReplyPreferences,
    ReplyResponse,
    ReviewCreateRequest,
    ReviewedReplyResponse,
    ReviewResponse,
    ReviseReplyRequest,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    # 이 라우터의 모든 주소 앞에 붙는 공통 주소 (Spring의 @RequestMapping("/api/reviews"))
    prefix="/api/reviews",
    # /docs 화면에서 API를 묶어 보여 줄 이름
    tags=["리뷰"],
)

preferences_router = APIRouter(prefix="/api/reply-preferences", tags=["답글 설정"])


# 핸들러가 언제, 어느 스레드에서 실행되는지 터미널에 찍는다.
def log(handler: str, message: str) -> None:
    now = time.strftime("%H:%M:%S")
    thread = threading.current_thread().name
    print(f"{now} [{handler}] {message} (스레드: {thread})")


# 1. 리뷰 등록 (def)
# 성공하면 201. 요청 본문이 ReviewCreateRequest의 조건에 맞지 않으면 핸들러를 실행하지 않고 422로 응답한다.
@router.post("", status_code=201, summary="리뷰 등록")
def register_review(request: ReviewCreateRequest, service: ReviewServiceDep) -> ReviewResponse:
    return service.register_review(request)

# 요청 : POST /api/reviews
# {"customerName": "최유진", "rating": 4, "content": "생갈치쉐이크가 의외로 달콤해서 놀랐어요."}
# 응답 : 201
# {"reviewId": 4, "customerName": "최유진", "rating": 4, "content": "생갈치쉐이크가 의외로 달콤해서 놀랐어요.", "reply": null}
#
# 요청 : POST /api/reviews {"customerName": "최유진", "rating": 6, "content": "맛있어요"}
# 응답 : 422 {"detail": [{"type": "less_than_equal", "loc": ["body", "rating"], "msg": "Input should be less than or equal to 5", ...}]}


# 2. 리뷰 전체 조회 (def)
@router.get("", summary="리뷰 전체 조회")
def find_reviews(service: ReviewServiceDep) -> list[ReviewResponse]:
    return service.find_reviews()


# 3. 리뷰 하나 조회 (def)
# 없는 번호면 서비스가 ReviewNotFoundError를 던지고, 예외 처리기가 404로 응답한다.
@router.get("/{review_id}", summary="리뷰 하나 조회")
def find_review(review_id: int, service: ReviewServiceDep) -> ReviewResponse:
    return service.find_review(review_id)

# 요청 : GET /api/reviews/99
# 응답 : 404 {"code": "ERROR_CODE_00001", "description": "조회 실패", "detail": "해당 리뷰가 존재하지 않습니다. reviewId=99"}
#
# 요청 : GET /api/reviews/abc
# 응답 : 422 (review_id가 int가 아니다. 핸들러는 실행되지 않는다)


# 4. 사장님 답글 초안 생성 (async def + await)
# 저장된 리뷰로 프롬프트를 만들어 LLM에 보내고, 응답을 기다리는 동안 이벤트 루프는 다른 요청을 처리한다.
@router.post(
    "/{review_id}/reply",
    summary="규정 근거가 있는 답글 초안 생성",
    responses={404: {"description": "리뷰 없음"}, 409: {"description": "동일 리뷰 처리 중"}, 502: {"description": "AI 응답 실패"}},
)
async def create_reply(review_id: int, service: ReplyServiceDep) -> ReviewedReplyResponse:

    log("reply", f"요청 받음 reviewId={review_id}")
    response = await service.create_reply_async(review_id)
    log("reply", f"응답 보냄 status={response.status}, attempts={response.attempts}")

    return response


@router.post(
    "/{review_id}/reply/revise",
    summary="직전 답글 수정",
    responses={
        404: {"description": "리뷰 없음"},
        409: {"description": "이전 답글 없음 또는 동일 리뷰 처리 중"},
        502: {"description": "AI 응답 실패"},
    },
)
async def revise_reply(
    review_id: int, payload: ReviseReplyRequest, service: ReplyServiceDep,
) -> ReviewedReplyResponse:
    return await service.create_reply_async(review_id, payload.instruction)


@router.post(
    "/{review_id}/reply/stream",
    response_class=EventSourceResponse,
    summary="답글 작성 과정 스트리밍",
    responses={404: {"description": "리뷰 없음"}, 409: {"description": "동일 리뷰 처리 중"}},
)
async def stream_reply(
    review: StreamReplyDep, service: ReplyServiceDep,
) -> AsyncIterator[ServerSentEvent]:
    try:
        async with aclosing(service.stream_reply(review)) as updates:
            async for event, payload in updates:
                if isinstance(payload, ReviewedReplyResponse):
                    payload = payload.model_dump(by_alias=True)
                yield ServerSentEvent(event=event, data=payload)
    except AiRequestError as error:
        yield ServerSentEvent(event="error", data={"code": "ERROR_CODE_00011", "detail": str(error)})
    except Exception as error:
        logger.error("답글 스트림 실패 review_id=%s (%s)", review.review_id, type(error).__name__)
        yield ServerSentEvent(
            event="error", data={"code": "STREAM_FAILED", "detail": "답글 처리가 중단됐습니다."},
        )


@preferences_router.get("", summary="매장 답글 선호 조회")
async def get_reply_preferences(service: ReplyServiceDep) -> ReplyPreferences:
    return await service.get_preferences()


@preferences_router.put("", summary="매장 답글 선호 저장")
async def put_reply_preferences(payload: ReplyPreferences, service: ReplyServiceDep) -> ReplyPreferences:
    return await service.put_preferences(payload)


# 5. 사장님 답글 초안 생성 - 동기 버전 (def + 동기 메서드)
# 비교용 LCEL 체인은 스레드 풀의 스레드 하나에서 모델 응답을 기다린다.
@router.post("/{review_id}/reply/sync", summary="사장님 답글 초안 생성 - 동기 버전 (def)")
def create_reply_sync(review_id: int, service: ReplyServiceDep) -> ReplyResponse:

    log("reply/sync", f"요청 받음 reviewId={review_id}")
    response = service.create_reply(review_id)
    log("reply/sync", "응답 보냄")

    return response
