"""
ReviewService 단위 테스트

웹 서버 없이 ReviewService 객체와 ReviewRepository 객체만으로 확인한다.
각 테스트는 준비(Given) -> 실행(When) -> 검증(Then) 순서로 적었다.
"""

import pytest

from app.core.exceptions import ReviewNotFoundError
from app.repositories.review_repository import ReviewRepository
from app.schemas.review import ReviewCreateRequest
from app.services.review_service import ReviewService


def test_find_reviews(repository: ReviewRepository):
    # 준비
    service = ReviewService(repository)

    # 실행
    reviews = service.find_reviews()

    # 검증 : 미리 준비한 리뷰 3개가 번호 순서대로 나온다.
    assert [review.review_id for review in reviews] == [1, 2, 3]
    assert reviews[0].customer_name == "김하늘"


def test_find_review(repository: ReviewRepository):
    service = ReviewService(repository)

    review = service.find_review(2)

    assert review.customer_name == "박준호"
    assert review.rating == 2
    # 답글 API를 부르기 전이므로 답글이 없다.
    assert review.reply is None


def test_find_review_not_found(repository: ReviewRepository):
    service = ReviewService(repository)

    # with 블록 안에서 ReviewNotFoundError가 발생해야 성공한다. (JUnit의 assertThrows)
    with pytest.raises(ReviewNotFoundError) as exc_info:
        service.find_review(99)

    assert exc_info.value.review_id == 99
    assert str(exc_info.value) == "해당 리뷰가 존재하지 않습니다. reviewId=99"


def test_register_review(repository: ReviewRepository):
    service = ReviewService(repository)
    # 파이썬 코드에서 DTO를 만들 때는 snake_case 이름으로 값을 넣는다. (populate_by_name=True)
    request = ReviewCreateRequest(customer_name="최유진", rating=4, content="생갈치쉐이크가 의외로 달콤해서 놀랐어요.")

    created = service.register_review(request)

    # 다음 번호(4)가 붙고, 저장소에서 다시 조회된다.
    assert created.review_id == 4
    assert created.reply is None
    assert service.find_review(4).content == "생갈치쉐이크가 의외로 달콤해서 놀랐어요."


def test_request_strips_whitespace():
    # 앞뒤 공백은 지운 뒤 저장한다. (ReviewCreateRequest의 str_strip_whitespace=True)
    request = ReviewCreateRequest(customer_name="  최유진 ", rating=4, content=" 맛있어요 ")

    assert request.customer_name == "최유진"
    assert request.content == "맛있어요"
