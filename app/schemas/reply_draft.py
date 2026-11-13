"""모델이 작성하는 초안. API 응답 DTO에는 서비스에서 필요한 값을 옮긴다."""

from typing import Literal

from pydantic import BaseModel, Field


class ReplyDraft(BaseModel):
    reply: str = Field(description="사장님이 고객에게 전달할 답글 초안 본문")
    tone: Literal["사과", "감사", "일반"] = Field(description="초안에서 사용한 주된 어조")
    needs_followup: bool = Field(description="환불 요청이나 이물질 신고처럼 매장이 직접 확인해야 하는지")


class GroundedReplyDraft(ReplyDraft):
    article_ids: list[str] = Field(description="답글에 실제로 반영한 운영 규정 조문 번호. 예: 제12조")


class ReplyVerdict(BaseModel):
    passed: bool
    reason: str = Field(description="통과하지 못한 경우 답글에서 고칠 부분")
