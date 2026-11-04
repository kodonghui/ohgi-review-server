"""
리뷰 라우터 : 리뷰 API와 사장님 답글 API의 핸들러 (Spring의 ReviewController 역할)

핸들러는 요청을 서비스에 넘기고 결과를 반환하기만 한다.
- def 핸들러       : 스레드 풀(기본 40개)의 스레드에서 실행된다. 터미널에 "AnyIO worker thread"로 찍힌다.
- async def 핸들러 : 이벤트 루프가 도는 스레드에서 실행된다. 터미널에 "MainThread"로 찍힌다.
                     await로 LLM을 기다리는 동안 이벤트 루프는 다른 요청을 처리한다.

답글 API는 LLM의 응답을 몇 초씩 기다리는 일이라 async def + await로 만들었다(POST /{review_id}/reply).
비교를 위해 def + 동기 메서드 버전도 하나 더 두었다(POST /{review_id}/reply/sync).
"""

import threading
import time

from fastapi import APIRouter

from app.dependencies import ReplyServiceDep, ReviewServiceDep
from app.schemas.review import ReplyResponse, ReviewCreateRequest, ReviewResponse

router = APIRouter(
    # 이 라우터의 모든 주소 앞에 붙는 공통 주소 (Spring의 @RequestMapping("/api/reviews"))
    prefix="/api/reviews",
    # /docs 화면에서 API를 묶어 보여 줄 이름
    tags=["리뷰"],
)


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
@router.post("/{review_id}/reply", summary="사장님 답글 초안 생성 (async def)")
async def create_reply(review_id: int, service: ReplyServiceDep) -> ReplyResponse:

    log("reply", f"요청 받음 reviewId={review_id}")
    response = await service.create_reply_async(review_id)
    log("reply", "응답 보냄")

    return response

# 요청 : POST /api/reviews/2/reply   (요청 본문 없음)
# 응답 : 200 (가짜 LLM으로 실행한 서버. 실제 OpenAI면 reply에 답글 문장이 들어간다)
# {"reviewId": 2, "reply": "(가짜 응답) 3.0초 기다렸습니다. 받은 프롬프트 : 손님 이름 : 박준호\n별점 : 2점 (5점 만점)\n리뷰 내용 :\n우럭스무디에서 ...", "model": "fake-llm"}
#
# 터미널
#   10:15:02 [reply] 요청 받음 reviewId=2 (스레드: MainThread)
#   10:15:05 [reply] 응답 보냄 (스레드: MainThread)


# 5. 사장님 답글 초안 생성 - 동기 버전 (def + 동기 메서드)
# 4번과 같은 일을 하지만, 스레드 풀의 스레드 하나가 LLM의 응답이 올 때까지 멈춰서 기다린다.
@router.post("/{review_id}/reply/sync", summary="사장님 답글 초안 생성 - 동기 버전 (def)")
def create_reply_sync(review_id: int, service: ReplyServiceDep) -> ReplyResponse:

    log("reply/sync", f"요청 받음 reviewId={review_id}")
    response = service.create_reply(review_id)
    log("reply/sync", "응답 보냄")

    return response

# 터미널
#   10:16:10 [reply/sync] 요청 받음 reviewId=2 (스레드: AnyIO worker thread)
#   10:16:13 [reply/sync] 응답 보냄 (스레드: AnyIO worker thread)


# [Postman으로 두 버전 비교하기]
# .env에 FAKE_LLM=true, FAKE_LLM_DELAY_SECONDS=5, THREAD_POOL_SIZE=1을 적고 서버를 다시 실행한다.
# (스레드 풀에 스레드가 하나뿐인 서버. 기다리는 시간이 길어야 차이가 잘 보인다)
# Postman에서 탭 두 개를 열고, 첫 번째 탭의 Send를 누른 직후 두 번째 탭의 Send를 누른다.
#   탭 A, 탭 B : 둘 다 POST /api/reviews/1/reply/sync
#   -> 탭 B는 10초 가까이 걸린다. 하나뿐인 스레드를 탭 A가 쓰고 있어서, 탭 A가 끝날 때까지 시작하지도 못했다.
#   탭 A, 탭 B : 둘 다 POST /api/reviews/1/reply
#   -> 둘 다 5초 만에 응답을 받는다. await로 기다리는 동안 이벤트 루프가 두 번째 요청도 받았다.
# 확인했으면 THREAD_POOL_SIZE 줄을 다시 지워 두자.
