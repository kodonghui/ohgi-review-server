"""
리뷰 엔티티 (Spring의 review/domain/Review.java 역할)

DB에 연결하지 않으므로 매핑 정보(@Entity, @Column) 없이 값만 가진 @dataclass로 만든다.
필드 이름은 snake_case로 적고, JSON의 camelCase는 DTO(schemas/review.py)가 맡는다.
"""

from dataclasses import dataclass


@dataclass
class Review:
    # 리뷰 번호(PK). 새 리뷰는 0으로 만들고, ReviewRepository가 저장할 때 1, 2, 3 ... 순서로 번호를 붙인다.
    review_id: int
    # 손님 이름
    customer_name: str
    # 별점 (1 ~ 5)
    rating: int
    # 리뷰 내용
    content: str
