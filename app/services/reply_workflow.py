"""규정 근거가 있는 답글을 작성하고, 검토에 실패하면 한 번 더 작성한다."""

import re
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from app.schemas.reply_draft import GroundedReplyDraft, ReplyVerdict
from app.schemas.review import ReplyPreferences
from app.services.policy_search import PolicySearchResult


MAX_DRAFT_ATTEMPTS = 2
GRAPH_RECURSION_LIMIT = 8
FOLLOWUP_ISSUE = re.compile(r"직원\s*확인")
COMPENSATION_PROMISE = re.compile(
    r"(?:환불|보상|쿠폰|재배달|재조리)(?:을|를|은)?\s*"
    r"(?:(?:바로|즉시)\s*)?"
    r"(?:해\s*드리겠습니다|드리겠습니다|진행하겠습니다|확정했습니다|지급하겠습니다|완료했습니다)"
)
DEADLINE_MENTION = re.compile(r"\d+(?:\.\d+)?\s*(?:시간|분\s*이내|일\s*이내)")
INCIDENT_CONTEXT = re.compile(
    r"오배송|다른\s*(?:메뉴|음식)|이물질|용기\s*파손|내용물\s*유출|기한|수령|사진|접수"
)
ALCOHOL_DELIVERY_PROMISE = re.compile(
    r"(?:주류|맥주|와인|라거|소주|막걸리|술)[^,.!?。]{0,20}(?:배달|배송|보내|가져다)"
    r"[^,.!?。]{0,12}(?:(?<!불)가능(?!하지\s*않)|됩니다|할\s*수\s*있|해\s*드리|드릴\s*수\s*있)"
)


def policy_notice_problem(reply: str, evidence: PolicySearchResult) -> str:
    """필수 문구를 사건별로 확인하고 별도로 덧붙인 접수 기한을 걸러낸다."""
    notices = [notice for item in evidence.evidence if (notice := item.required_notice())]
    if not notices:
        return ""
    text = " ".join(unicodedata.normalize("NFC", reply).split())
    for notice in notices:
        body = notice.removesuffix(".")
        match = re.search(rf"(?:^|(?<=[ ,.!?。])){re.escape(body)}(?=$|[.,!?。])", text)
        if match is None:
            return f"확인한 안내를 빠짐없이 써 주세요. {notice}"
        text = text[:match.start()] + text[match.end():]
    clauses = re.split(r"[,!?。]|\.(?=\s|$)", text)
    if evidence.required_hours and any(
        DEADLINE_MENTION.search(clause) and INCIDENT_CONTEXT.search(clause) for clause in clauses
    ):
        return "접수 기한은 확인된 사건별 안내 문구에만 넣어 주세요."
    if any(item.topic == "alcohol_delivery" for item in evidence.evidence) and ALCOHOL_DELIVERY_PROMISE.search(text):
        return "주류 배달 여부는 확인된 안내 문구대로 써 주세요."
    return ""


class ReplyWorkflowInput(TypedDict):
    customer_name: str
    rating: int
    review_text: str
    preferences: NotRequired[ReplyPreferences]
    previous_reply: NotRequired[str | None]
    revision_instruction: NotRequired[str | None]
    stream_tokens: NotRequired[bool]
    next_reply_version: NotRequired[int]


class ReplyWorkflowOutput(TypedDict):
    status: Literal["ready", "needs_review"]
    draft: GroundedReplyDraft | None
    attempts: int
    feedback: str


class ReplyWorkflowState(ReplyWorkflowInput):
    evidence: PolicySearchResult
    draft: GroundedReplyDraft | None
    attempts: int
    feedback: str
    status: Literal["ready", "needs_review"] | None
    last_accepted_reply: NotRequired[str]
    last_accepted_version: NotRequired[int]


@dataclass(frozen=True)
class ReplyModelCalls:
    draft: Callable[[ReplyWorkflowState], Awaitable[GroundedReplyDraft]]
    review: Callable[[ReplyWorkflowState], Awaitable[ReplyVerdict]]


def build_reply_workflow(
    retrieve: Callable[[str], Awaitable[PolicySearchResult]],
    *,
    checkpointer: InMemorySaver | None = None,
) -> CompiledStateGraph:
    """작성과 검토는 외부에서 주입한다. 저장에는 통과한 결과만 넘긴다."""

    async def retrieve_evidence(state: ReplyWorkflowInput) -> dict:
        get_stream_writer()({"event": "progress", "stage": "retrieve"})
        evidence = await retrieve(state["review_text"])
        initial = {
            "evidence": evidence,
            "attempts": 0,
            "feedback": "",
            "draft": None,
            "status": None,
        }
        if evidence.needs_manual_review:
            initial.update(status="needs_review", feedback="적용할 규정을 매장에서 확인해 주세요.")
        if evidence.reason or evidence.retried_questions:
            get_stream_writer()({
                "event": "progress", "stage": "evidence",
                "reason": evidence.reason or "evidence_recovered",
                "retriedQuestions": list(evidence.retried_questions),
            })
        return initial

    def after_retrieval(state: ReplyWorkflowState) -> Literal["draft", "__end__"]:
        return END if state["evidence"].needs_manual_review else "draft"

    async def draft_reply(state: ReplyWorkflowState, runtime: Runtime[ReplyModelCalls]) -> dict:
        attempt = state["attempts"] + 1
        get_stream_writer()({"event": "progress", "stage": "draft", "attempt": attempt})
        return {"draft": await runtime.context.draft(state), "attempts": attempt}

    async def review_reply(state: ReplyWorkflowState, runtime: Runtime[ReplyModelCalls]) -> dict:
        get_stream_writer()({"event": "progress", "stage": "review", "attempt": state["attempts"]})
        draft = state["draft"]
        evidence = state["evidence"]
        if draft is None or not draft.reply.strip():
            feedback = "빈 답글을 작성했습니다."
        elif (
            not set(evidence.required_ids).issubset(draft.article_ids)
            or not set(draft.article_ids).issubset(article.id for article in evidence.articles)
        ):
            feedback = "필요한 조문을 모두 반영하고 확인한 조문만 표시해 주세요."
        elif (evidence.requires_followup or FOLLOWUP_ISSUE.search(state["review_text"])) and not draft.needs_followup:
            feedback = "매장이 직접 확인해야 하는 내용을 표시해 주세요."
        elif evidence.articles and not draft.article_ids:
            feedback = "안내를 뒷받침하는 조문을 확인하고 표시해 주세요."
        elif COMPENSATION_PROMISE.search(draft.reply):
            feedback = "확인하기 전에는 환불이나 보상을 약속할 수 없습니다."
        elif issue := policy_notice_problem(draft.reply, evidence):
            feedback = issue
        else:
            verdict = await runtime.context.review(state)
            if verdict.passed:
                return {
                    "status": "ready",
                    "feedback": "",
                    "last_accepted_reply": draft.reply,
                    "last_accepted_version": state.get("next_reply_version", 1),
                }
            feedback = verdict.reason.strip() or "리뷰와 규정 적용 조건을 다시 확인해 주세요."

        get_stream_writer()({
            "event": "progress", "stage": "review_failed", "attempt": state["attempts"],
        })
        if state["attempts"] >= MAX_DRAFT_ATTEMPTS:
            return {"status": "needs_review", "draft": None, "feedback": feedback}
        return {"feedback": feedback}

    def after_review(state: ReplyWorkflowState) -> Literal["draft", "__end__"]:
        return END if state["status"] else "draft"

    graph = StateGraph(
        ReplyWorkflowState,
        input_schema=ReplyWorkflowInput,
        output_schema=ReplyWorkflowOutput,
        context_schema=ReplyModelCalls,
    )
    graph.add_node("retrieve", retrieve_evidence)
    graph.add_node("draft", draft_reply)
    graph.add_node("review", review_reply)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", after_retrieval)
    graph.add_edge("draft", "review")
    graph.add_conditional_edges("review", after_review)
    return graph.compile(checkpointer=checkpointer).with_config({"recursion_limit": GRAPH_RECURSION_LIMIT})
