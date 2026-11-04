"""
리뷰 저장소 (Spring의 ReviewRepository 역할)

DB 대신 딕셔너리에 저장한다. 키는 리뷰 번호, 값은 Review 객체다.
서버를 다시 실행하면 처음 상태(미리 준비한 리뷰 3개)로 돌아간다.
ReviewRepository 객체는 서버 전체에 하나만 있어야 하고, 하나만 만들어 두는 일은 dependencies.py에서 한다.
"""

from app.domain.review import Review


# 서버가 실행될 때 미리 넣어 둘 리뷰. 번호는 모두 0으로 두고, 저장할 때 1번부터 붙인다.
# 리스트를 변수에 담지 않고 함수로 만든 이유 : ReviewRepository 객체를 만들 때마다(테스트마다) 새 Review 객체로 시작하기 위해서다.
def init_reviews() -> list[Review]:
    return [
        Review(0, "김하늘", 5, "열무김치라떼가 생각보다 고소하고 맛있었어요. 직원분들도 친절해서 기분 좋게 먹고 갑니다."),
        Review(0, "박준호", 2, "우럭스무디에서 비린 맛이 나서 반 넘게 남겼어요. 주문하고 음료가 나올 때까지 20분이나 걸렸습니다."),
        Review(0, "이서연", 3, "갈릭미역파르페는 신기하고 맛도 괜찮았는데 마늘 칩이 눅눅해서 아쉬웠어요."),
    ]


class ReviewRepository:

    def __init__(self):
        # 리뷰 번호를 키로, Review 객체를 값으로 관리한다. (딕셔너리는 저장한 순서를 유지한다)
        self._reviews: dict[int, Review] = {}

        # 다음에 저장할 새 리뷰에 붙일 번호 (AUTO_INCREMENT)
        self._next_id = 1

        # 미리 준비한 리뷰를 저장한다.
        for review in init_reviews():
            self.save(review)

    # 1. 전체 리뷰 조회
    def find_all(self) -> list[Review]:
        """
        저장된 리뷰를 모두 조회한다.

        Returns:
            list[Review]: 리뷰 번호 순서대로 담은 새 리스트
        """

        return list(self._reviews.values())

    # 2. 리뷰 하나 조회
    def find_by_id(self, review_id: int) -> Review | None:
        """
        리뷰 번호로 리뷰 하나를 조회한다.

        Args:
            review_id (int): 조회할 리뷰 번호

        Returns:
            Review | None: 찾은 리뷰. 없으면 None
        """

        # 없는 번호일 때 KeyError 대신 None을 반환하도록 get()을 사용한다. (JPA findById()의 Optional 역할)
        return self._reviews.get(review_id)

    # 3. 리뷰 저장
    def save(self, review: Review) -> Review:
        """
        리뷰를 저장한다. 번호가 0인 새 리뷰에는 다음 번호를 붙인다.

        Args:
            review (Review): 저장할 리뷰

        Returns:
            Review: 번호가 붙은 리뷰 (전달받은 객체와 같은 객체)
        """

        # 새 리뷰면 번호를 붙이고, 다음 번호를 하나 올린다. (JPA의 IDENTITY 전략을 흉내 냄)
        if review.review_id == 0:
            review.review_id = self._next_id
            self._next_id += 1

        # 같은 번호의 리뷰가 이미 있으면 값을 덮어쓴다. (=수정. Spring Data JPA의 save()와 같다)
        self._reviews[review.review_id] = review

        return review
