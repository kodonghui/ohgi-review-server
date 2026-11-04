"""
예외와 예외 처리기 (Spring의 사용자 정의 예외 + @ControllerAdvice 역할)

- ReviewNotFoundError : 없는 리뷰를 조회하거나, 없는 리뷰에 답글을 요청했을 때 -> 404
- AiRequestError      : OpenAI 호출이 실패했거나, 답글로 쓸 수 없는 응답이 왔을 때 -> 502

서비스는 예외를 던지기만 하고, 예외를 응답으로 바꾸는 일은 여기 등록한 처리기가 한다.
오류 응답은 chap02와 같은 {code, description, detail} 형태다.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


# 리뷰를 찾지 못했을 때 서비스가 던지는 예외 (Spring 프로젝트의 NotFoundException)
class ReviewNotFoundError(Exception):

    def __init__(self, review_id: int):
        super().__init__(f"해당 리뷰가 존재하지 않습니다. reviewId={review_id}")
        self.review_id = review_id


# OpenAI 호출이 실패했을 때 LLM 클라이언트가 던지는 예외. 메시지는 예외를 만들 때 전달한다.
# 예: AiRequestError("OpenAI 서버에 연결할 수 없습니다.")
class AiRequestError(Exception):
    pass


# main.py에서 app을 넘겨 호출한다. 이 함수 안에 정의한 처리기들이 app에 등록된다.
def register_exception_handlers(app: FastAPI) -> None:

    # 서비스가 ReviewNotFoundError를 던지면 FastAPI가 이 함수를 호출하고, 반환한 응답을 보낸다.
    # 핸들러(라우터의 함수)는 예외를 잡지 않아도 된다.
    @app.exception_handler(ReviewNotFoundError)
    def handle_review_not_found(request: Request, e: ReviewNotFoundError):
        return JSONResponse(
            status_code=404,
            content={
                "code": "ERROR_CODE_00001",
                "description": "조회 실패",
                # str(e)는 예외를 만들 때 전달한 메시지다. (Java의 e.getMessage())
                "detail": str(e),
            },
        )

    # 502 Bad Gateway : "우리 서버가 호출한 다른 서버(OpenAI)가 실패했다"는 뜻이다. 우리 서버의 잘못(500)과 구분한다.
    @app.exception_handler(AiRequestError)
    def handle_ai_request_error(request: Request, e: AiRequestError):
        return JSONResponse(
            status_code=502,
            content={
                "code": "ERROR_CODE_00011",
                "description": "AI 응답 실패",
                "detail": str(e),
            },
        )

# 요청 : GET /api/reviews/99
# 응답 : 404 {"code": "ERROR_CODE_00001", "description": "조회 실패", "detail": "해당 리뷰가 존재하지 않습니다. reviewId=99"}
#
# 요청 : POST /api/reviews/1/reply   (OPENAI_API_KEY에 잘못된 키를 넣고 실행한 서버)
# 응답 : 502 {"code": "ERROR_CODE_00011", "description": "AI 응답 실패", "detail": "OpenAI API 키가 올바르지 않습니다. OPENAI_API_KEY를 확인하세요."}
