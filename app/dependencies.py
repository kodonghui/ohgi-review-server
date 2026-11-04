"""
의존성 함수 (Spring의 빈 등록 + 생성자 주입 역할)

핸들러의 매개변수에 Depends(의존성 함수)를 적어 두면, 요청이 들어올 때마다 FastAPI가 의존성 함수를 호출해서 반환값을 넣어 준다.
- ReviewRepository : 서버 전체에 하나만 있어야 하므로 이 모듈에서 한 번 만들어 두고 같은 객체를 반환한다.
- ReviewService    : 요청마다 새로 만든다. 저장소만 들고 있는 가벼운 객체라서 괜찮다.
"""

from typing import Annotated

from fastapi import Depends

from app.repositories.review_repository import ReviewRepository
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
