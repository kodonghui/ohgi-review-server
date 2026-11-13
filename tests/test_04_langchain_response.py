"""공급자의 실패 응답이 저장할 답글로 넘어가지 않는지 확인한다."""

import pytest
from langchain_core.messages import AIMessage

from app.clients.langchain_response import require_complete_message, require_reply_text
from app.core.exceptions import AiRequestError


def test_complete_message_passes_to_parser():
    message = AIMessage(
        content=[{"type": "text", "text": "리뷰를 남겨 주셔서 감사합니다."}],
        response_metadata={"status": "completed"},
    )
    assert require_complete_message(message) is message
    assert require_reply_text(" 리뷰를 남겨 주셔서 감사합니다.\n") == "리뷰를 남겨 주셔서 감사합니다."


@pytest.mark.parametrize(
    "message, reason",
    [
        (
            AIMessage(
                content="중간에 잘린 문장",
                response_metadata={
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
            ),
            "출력 토큰 상한",
        ),
        (AIMessage(content="중간에 잘린 문장", response_metadata={"finish_reason": "length"}), "출력 토큰 상한"),
        (AIMessage(content="", response_metadata={"finish_reason": "content_filter"}), "content_filter"),
        (AIMessage(content="", response_metadata={"status": "failed"}), "완성되지 않았습니다"),
        (
            AIMessage(content=[{"type": "refusal", "refusal": "거절"}], response_metadata={"status": "completed"}),
            "생성을 거절했습니다",
        ),
    ],
    ids=["responses-incomplete", "chat-completions-length", "content-filter", "failed", "refusal"],
)
def test_unusable_message_fails_before_parsing(message: AIMessage, reason: str):
    with pytest.raises(AiRequestError, match=reason):
        require_complete_message(message)


def test_blank_reply_is_rejected_after_parsing():
    with pytest.raises(AiRequestError, match="답글 문장이 없습니다"):
        require_reply_text(" \n ")
