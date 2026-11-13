"""LangChain 응답을 답글 체인에서 파싱하기 전에 확인한다."""

from langchain_core.messages import AIMessage

from app.core.exceptions import AiRequestError


def require_complete_message(message: AIMessage, *, expect_completed: bool = False) -> AIMessage:
    """응답 상태와 거절을 확인한 뒤 원래 메시지를 반환한다."""
    status = message.response_metadata.get("status")
    finish_reason = message.response_metadata.get("finish_reason")
    details = message.response_metadata.get("incomplete_details")
    reason = details.get("reason") if isinstance(details, dict) else None

    if reason == "max_output_tokens" or finish_reason == "length":
        raise AiRequestError("OpenAI 응답이 출력 토큰 상한에 걸려 중간에 끊겼습니다. OPENAI_MAX_OUTPUT_TOKENS를 늘리세요.")
    if reason == "content_filter" or finish_reason == "content_filter":
        raise AiRequestError("OpenAI 응답이 완성되지 않았습니다. (이유: content_filter)")
    if (expect_completed or status is not None) and status != "completed":
        raise AiRequestError("OpenAI 응답이 완성되지 않았습니다.")

    blocks = message.content if isinstance(message.content, list) else []
    if message.additional_kwargs.get("refusal") or any(
        isinstance(block, dict) and block.get("type") == "refusal" for block in blocks
    ):
        raise AiRequestError("OpenAI가 답글 생성을 거절했습니다.")
    return message


def require_reply_text(text: str) -> str:
    reply = text.strip()
    if not reply:
        raise AiRequestError("OpenAI 응답에 답글 문장이 없습니다.")
    return reply
