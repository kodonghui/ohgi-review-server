"""
리뷰 라우터 : 리뷰 API의 핸들러 (Spring의 ReviewController 역할)

핸들러는 요청을 서비스에 넘기고 결과를 반환하기만 한다.
리뷰 등록과 조회는 기다릴 일이 없는 짧은 작업이라 def 핸들러로 만들었다. def 핸들러는 스레드 풀의 스레드에서 실행된다.
"""

from fastapi import APIRouter

from app.dependencies import ReviewServiceDep
from app.schemas.review import ReviewCreateRequest, ReviewResponse

router = APIRouter(
    # 이 라우터의 모든 주소 앞에 붙는 공통 주소 (Spring의 @RequestMapping("/api/reviews"))
    prefix="/api/reviews",
    # /docs 화면에서 API를 묶어 보여 줄 이름
    tags=["리뷰"],
)


# 1. 리뷰 등록 (def)
# 성공하면 201. 요청 본문이 ReviewCreateRequest의 조건에 맞지 않으면 핸들러를 실행하지 않고 422로 응답한다.
@router.post("", status_code=201, summary="리뷰 등록")
def register_review(request: ReviewCreateRequest, service: ReviewServiceDep) -> ReviewResponse:
    return service.register_review(request)

# 요청 : POST /api/reviews
# {"customerName": "최유진", "rating": 4, "content": "생갈치쉐이크가 의외로 달콤해서 놀랐어요."}
# 응답 : 201
# {"reviewId": 4, "customerName": "최유진", "rating": 4, "content": "생갈치쉐이크가 의외로 달콤해서 놀랐어요."}
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
