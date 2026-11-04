"""
리뷰 API 테스트

TestClient로 HTTP 요청을 보내고 상태 코드와 JSON 본문을 확인한다.
라우터, 의존성 함수, 서비스, 저장소, 예외 처리기가 함께 잘 조립되었는지 확인하는 테스트다.
"""

import pytest
from fastapi.testclient import TestClient


def test_register_review(client: TestClient):
    # 요청 본문은 camelCase 키로 보낸다.
    body = {"customerName": "최유진", "rating": 4, "content": "생갈치쉐이크가 의외로 달콤해서 놀랐어요."}

    response = client.post("/api/reviews", json=body)

    # 201과 번호가 붙은 리뷰. 응답 키도 camelCase다.
    assert response.status_code == 201
    assert response.json() == {
        "reviewId": 4,
        "customerName": "최유진",
        "rating": 4,
        "content": "생갈치쉐이크가 의외로 달콤해서 놀랐어요.",
        "reply": None,
    }
    # 등록한 리뷰를 다시 조회할 수 있다.
    assert client.get("/api/reviews/4").json()["customerName"] == "최유진"


# 조건에 맞지 않는 요청 본문과, 422 응답의 loc가 가리켜야 하는 필드
@pytest.mark.parametrize(
    "body, wrong_field",
    [
        ({"customerName": "최유진", "rating": 6, "content": "맛있어요"}, "rating"),
        ({"customerName": "최유진", "rating": 0, "content": "맛있어요"}, "rating"),
        ({"customerName": "최유진", "rating": 4.5, "content": "맛있어요"}, "rating"),
        ({"customerName": "최유진", "content": "맛있어요"}, "rating"),
        ({"customerName": "", "rating": 4, "content": "맛있어요"}, "customerName"),
        ({"customerName": "최유진", "rating": 4, "content": "   "}, "content"),
        ({"customerName": "최유진", "rating": 4, "content": "가" * 501}, "content"),
    ],
    ids=["rating-6", "rating-0", "rating-4.5", "rating-missing", "name-empty", "content-blank", "content-501"],
)
def test_register_review_invalid(client: TestClient, body: dict, wrong_field: str):
    response = client.post("/api/reviews", json=body)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", wrong_field]
    # 잘못된 요청은 저장되지 않는다.
    assert len(client.get("/api/reviews").json()) == 3


def test_find_reviews(client: TestClient):
    response = client.get("/api/reviews")

    assert response.status_code == 200
    body = response.json()
    assert [review["reviewId"] for review in body] == [1, 2, 3]
    # 응답 키는 camelCase다. snake_case 키는 없다.
    assert "customer_name" not in body[0]


def test_find_review(client: TestClient):
    response = client.get("/api/reviews/1")

    assert response.status_code == 200
    assert response.json()["customerName"] == "김하늘"
    assert response.json()["reply"] is None


def test_find_review_not_found(client: TestClient):
    response = client.get("/api/reviews/99")

    # ReviewNotFoundError를 예외 처리기가 404 응답으로 바꿨다.
    assert response.status_code == 404
    assert response.json() == {
        "code": "ERROR_CODE_00001",
        "description": "조회 실패",
        "detail": "해당 리뷰가 존재하지 않습니다. reviewId=99",
    }


def test_find_review_wrong_path(client: TestClient):
    response = client.get("/api/reviews/abc")

    # 경로 변수가 int가 아니면 핸들러를 실행하지 않고 422로 응답한다.
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["path", "review_id"]
