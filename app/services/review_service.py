"""
리뷰 서비스 : 리뷰 등록과 조회 (Spring의 ReviewService 역할)

- 생성자로 ReviewRepository 객체를 주입받는다. 객체를 만들어 넣어 주는 일은 dependencies.py가 한다.
- FastAPI를 import 하지 않는 평범한 클래스라서, 웹 서버 없이도 이 클래스만 따로 테스트할 수 있다.
- 리뷰가 없으면 ReviewNotFoundError를 던진다. 이 예외를 404 응답으로 바꾸는 일은 예외 처리기(core/exceptions.py)가 한다.
"""

from app.core.exceptions import ReviewNotFoundError
from app.domain.review import Review
from app.repositories.review_repository import ReviewRepository
from app.schemas.review import ReviewCreateRequest, ReviewResponse


class ReviewService:

    def __init__(self, repository: ReviewRepository):
        self.repository = repository

    # 1. 전체 리뷰 조회
    def find_reviews(self) -> list[ReviewResponse]:
        """
        리뷰를 모두 조회한다.

        Returns:
            list[ReviewResponse]: 리뷰 번호 순서대로 담은 응답 DTO 목록
        """

        reviews = self.repository.find_all()

        return [ReviewResponse.from_review(review) for review in reviews]

    # 2. 리뷰 하나 조회
    def find_review(self, review_id: int) -> ReviewResponse:
        """
        리뷰 번호로 리뷰 하나를 조회한다.

        Args:
            review_id (int): 조회할 리뷰 번호

        Returns:
            ReviewResponse: 찾은 리뷰의 응답 DTO

        Raises:
            ReviewNotFoundError: 그 번호의 리뷰가 없을 때 (예외 처리기가 404로 응답한다)
        """

        review = self.repository.find_by_id(review_id)

        # 없으면 예외를 던진다. (Java의 orElseThrow())
        if review is None:
            raise ReviewNotFoundError(review_id)

        return ReviewResponse.from_review(review)

    # 3. 리뷰 등록
    def register_review(self, request: ReviewCreateRequest) -> ReviewResponse:
        """
        새 리뷰를 등록한다.

        Args:
            request (ReviewCreateRequest): 요청 본문. 값의 조건은 FastAPI가 이미 검사했다.

        Returns:
            ReviewResponse: 번호가 붙은 새 리뷰의 응답 DTO
        """

        # 요청 DTO의 값으로 엔티티를 만든다. 번호는 아직 없으므로 0이다.
        review = Review(
            review_id=0,
            customer_name=request.customer_name,
            rating=request.rating,
            content=request.content,
        )

        # 저장하면 ReviewRepository가 번호를 붙인다.
        saved = self.repository.save(review)

        return ReviewResponse.from_review(saved)
