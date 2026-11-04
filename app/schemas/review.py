"""
리뷰 요청, 응답 DTO와 답글 응답 DTO (Spring의 review/dto 패키지 역할)

- ReviewCreateRequest : POST /api/reviews의 요청 본문. Field 조건에 맞지 않으면 FastAPI가 핸들러를 실행하지 않고 422로 응답한다.
- ReviewResponse      : 리뷰 등록, 조회 API의 응답
- ReplyResponse       : 답글 API의 응답

엔티티(Review)를 그대로 응답하지 않고 응답 DTO로 바꿔서 응답한다.
요청 DTO에는 review_id가 없다. 번호는 저장소가 붙이기 때문이다.
"""

from pydantic import ConfigDict, Field

from app.domain.review import Review
from app.schemas.base import CamelModel


class ReviewCreateRequest(CamelModel):
    """리뷰 등록 요청 DTO. POST /api/reviews의 요청 본문이다."""

    # 문자열의 앞뒤 공백을 지운 뒤 조건을 검사한다. 공백만 보낸 이름이나 내용도 422가 된다. (Spring의 @NotBlank)
    # 부모(CamelModel)의 camelCase 설정은 그대로 물려받고, 이 설정이 더해진다.
    model_config = ConfigDict(str_strip_whitespace=True)

    # 손님 이름 : 1글자 이상 20글자 이하
    customer_name: str = Field(min_length=1, max_length=20)
    # 별점 : 1 이상 5 이하의 정수. 6이나 0, 4.5를 보내면 422
    rating: int = Field(ge=1, le=5)
    # 리뷰 내용 : 1글자 이상 500글자 이하
    content: str = Field(min_length=1, max_length=500)


class ReviewResponse(CamelModel):
    """리뷰 응답 DTO. 리뷰 등록, 조회 API가 모두 이 형태로 응답한다."""

    review_id: int
    customer_name: str
    rating: int
    content: str

    # 엔티티를 응답 DTO로 바꾸는 정적 팩토리 메서드 (Spring의 ReviewResponse.from(review))
    @classmethod
    def from_review(cls, review: Review) -> ReviewResponse:
        """
        Review 엔티티를 ReviewResponse로 바꾼다.

        Args:
            review (Review): ReviewRepository 객체에서 꺼낸 리뷰

        Returns:
            ReviewResponse: 응답으로 보낼 DTO
        """

        return cls(
            review_id=review.review_id,
            customer_name=review.customer_name,
            rating=review.rating,
            content=review.content,
        )


class ReplyResponse(CamelModel):
    """사장님 답글 초안 응답 DTO. POST /api/reviews/{review_id}/reply가 이 형태로 응답한다."""

    # 답글을 단 리뷰의 번호
    review_id: int
    # LLM이 만든 답글 초안
    reply: str
    # 답글을 만든 모델 이름. 가짜 LLM 클라이언트면 "fake-llm"이다. (가짜 서버인지 확인할 때 쓴다)
    model: str
