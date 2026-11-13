"""임시 Chroma 인덱스와 가짜 모델로 규정 검색과 재작성 경로를 검증한다."""

import asyncio
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings
from langchain_chroma import Chroma

from app.schemas.reply_draft import GroundedReplyDraft, ReplyVerdict
from app.services.policy_search import PolicySearch
from app.services.policy_topics import DELIVERY_ARTICLE, QUALITY_ARTICLE
from app.services.reply_workflow import ReplyModelCalls, ReplyWorkflowState, build_reply_workflow


REVIEW = {
    "customer_name": "연습 손님",
    "rating": 1,
    "review_text": "배달 용기가 깨져 있었고 주문한 메뉴와 다른 음식도 왔어요. 환불해 주세요.",
}
ARTICLE_IDS = [QUALITY_ARTICLE, DELIVERY_ARTICLE]


@pytest.fixture
def policy_search(tmp_path: Path) -> PolicySearch:
    client = chromadb.PersistentClient(
        path=str(tmp_path),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.create_collection("ohgi_regulations")
    collection.add(
        ids=ARTICLE_IDS,
        documents=[
            "다른 메뉴가 나온 경우에는 수령 후 24시간 안에 사진과 함께 알려야 한다. "
            "맛이 입에 맞지 않는다는 이유만으로는 교환이나 환불을 하지 않는다.",
            "배달 용기가 파손된 경우에는 수령 후 1시간 안에 사진과 함께 알려야 한다.",
        ],
        metadatas=[{"article_no": article_id} for article_id in ARTICLE_IDS],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    index = Chroma(client=client, collection_name="ohgi_regulations", create_collection_if_not_exists=False)
    return PolicySearch(index)


def test_retrieves_both_articles_for_one_review(policy_search: PolicySearch):
    evidence = asyncio.run(policy_search.retrieve(REVIEW["review_text"]))
    assert list(evidence.required_ids) == ARTICLE_IDS
    assert [article.id for article in evidence.articles] == ARTICLE_IDS
    assert evidence.required_hours == ("24", "1")
    assert evidence.needs_manual_review is False


def test_taste_complaint_retrieves_quality_article_when_refund_is_requested(policy_search: PolicySearch):
    complaint = asyncio.run(policy_search.retrieve("맛이 입에 맞지 않았어요."))
    request = asyncio.run(policy_search.retrieve("맛이 입에 맞지 않았어요. 환불해 주세요."))

    assert complaint.required_ids == ()
    assert complaint.needs_manual_review is False
    assert request.required_ids == (QUALITY_ARTICLE,)
    assert [article.id for article in request.articles] == [QUALITY_ARTICLE]


def test_coupon_praise_can_receive_a_regular_draft(policy_search: PolicySearch):
    evidence = asyncio.run(policy_search.retrieve("쿠폰으로 할인받고 맛있게 먹었습니다."))
    assert evidence.required_ids == ()
    assert evidence.needs_manual_review is False


def test_missing_required_article_goes_to_staff(
    policy_search: PolicySearch, monkeypatch: pytest.MonkeyPatch
):
    async def no_documents(index, ids):
        assert ids == (DELIVERY_ARTICLE,)
        return []

    monkeypatch.setattr(Chroma, "aget_by_ids", no_documents)

    async def unexpected_model_call(state):
        pytest.fail("필요한 조문이 누락됐는데 모델을 호출했습니다.")

    graph = build_reply_workflow(policy_search.retrieve)
    result = asyncio.run(graph.ainvoke(
        {**REVIEW, "review_text": "배달 용기가 깨졌습니다."},
        context=ReplyModelCalls(unexpected_model_call, unexpected_model_call),
    ))
    assert result["status"] == "needs_review"
    assert result["attempts"] == 0
    assert result["draft"] is None


def test_draft_is_rewritten_with_feedback_and_reuses_evidence(policy_search: PolicySearch):
    generation_feedback = []
    searched = []

    async def retrieve(review_text):
        searched.append(review_text)
        return await policy_search.retrieve(review_text)

    async def draft(state: ReplyWorkflowState):
        generation_feedback.append(state["feedback"])
        reply = "사진을 보내 주시면 매장에서 확인하겠습니다."
        if state["feedback"]:
            reply = " ".join(item.required_notice() for item in state["evidence"].evidence)
        return GroundedReplyDraft(
            reply=reply,
            tone="사과",
            needs_followup=True,
            article_ids=list(state["evidence"].required_ids),
        )

    async def review(state: ReplyWorkflowState):
        draft = state["draft"]
        assert draft is not None
        contains_both_deadlines = all(item.required_notice() in draft.reply for item in state["evidence"].evidence)
        return ReplyVerdict(passed=contains_both_deadlines, reason="오배송과 파손 안내를 각각 써 주세요.")

    graph = build_reply_workflow(retrieve)
    result = asyncio.run(graph.ainvoke(REVIEW, context=ReplyModelCalls(draft, review)))
    assert result["status"] == "ready"
    assert result["attempts"] == 2
    draft = result["draft"]
    assert draft is not None
    assert draft.article_ids == ARTICLE_IDS
    assert generation_feedback == ["", "확인한 안내를 빠짐없이 써 주세요. "
                                       "다른 메뉴나 이물질은 수령 후 24시간 이내에 사진과 함께 알려 주세요."]
    assert searched == [REVIEW["review_text"]]


@pytest.mark.parametrize("article_ids", [[QUALITY_ARTICLE], [*ARTICLE_IDS, "제99조"]], ids=["missing", "unsupported"])
def test_unverified_draft_is_withheld_at_attempt_limit(policy_search: PolicySearch, article_ids: list[str]):
    async def draft(state):
        return GroundedReplyDraft(
            reply="이 문제를 확인해 보겠습니다.",
            tone="사과",
            needs_followup=True,
            article_ids=article_ids,
        )

    async def review(state):
        pytest.fail("조문 검사에 실패한 답글을 모델 판정으로 넘겼습니다.")

    graph = build_reply_workflow(policy_search.retrieve)
    result = asyncio.run(graph.ainvoke(REVIEW, context=ReplyModelCalls(draft, review)))
    assert result["status"] == "needs_review"
    assert result["attempts"] == 2
    assert result["draft"] is None


@pytest.mark.parametrize(
    "review_text",
    ["이벤트 가격을 보상해 주세요.", "배달 용기가 깨져 있었고 쿠폰을 발급해 주세요."],
)
def test_unsupported_compensation_request_goes_to_staff(policy_search: PolicySearch, review_text: str):
    async def unexpected_model_call(state):
        pytest.fail("적용 규정을 확인할 수 없는 요청에서 모델을 호출했습니다.")

    graph = build_reply_workflow(policy_search.retrieve)
    result = asyncio.run(graph.ainvoke(
        {**REVIEW, "review_text": review_text},
        context=ReplyModelCalls(unexpected_model_call, unexpected_model_call),
    ))
    assert result["status"] == "needs_review"
    assert result["attempts"] == 0
    assert result["draft"] is None


def test_second_failed_verdict_withholds_draft(policy_search: PolicySearch):
    reviews = []

    async def draft(state: ReplyWorkflowState):
        return GroundedReplyDraft(
            reply=" ".join(item.required_notice() for item in state["evidence"].evidence),
            tone="사과",
            needs_followup=True,
            article_ids=ARTICLE_IDS,
        )

    async def review(state: ReplyWorkflowState):
        reviews.append(state["attempts"])
        return ReplyVerdict(passed=False, reason="두 조문의 접수 기한을 안내해 주세요.")

    graph = build_reply_workflow(policy_search.retrieve)
    result = asyncio.run(graph.ainvoke(REVIEW, context=ReplyModelCalls(draft, review)))
    assert reviews == [1, 2]
    assert result["status"] == "needs_review"
    assert result["draft"] is None
    assert result["feedback"] == "두 조문의 접수 기한을 안내해 주세요."


@pytest.mark.parametrize("reply", [
    "오배송 사진은 수령 1시간 안에, 용기 파손 사진은 수령 24시간 안에 알려 주세요.",
    "다른 메뉴나 이물질은 수령 후 24시간 이내에 사진과 함께 알려 주세요. "
    "용기 파손이나 내용물 유출은 수령 후 1시간 이내에 사진과 함께 알려 주세요. 오배송은 1시간 안에 알려 주세요.",
])
def test_swapped_or_added_deadlines_never_reach_the_model_judge(policy_search: PolicySearch, reply: str):
    async def draft(_state):
        return GroundedReplyDraft(reply=reply, tone="사과", needs_followup=True, article_ids=ARTICLE_IDS)

    async def should_not_run(_state):
        pytest.fail("잘못된 사건별 기한을 모델 판정으로 넘겼습니다.")

    graph = build_reply_workflow(policy_search.retrieve)
    result = asyncio.run(graph.ainvoke(REVIEW, context=ReplyModelCalls(draft, should_not_run)))
    assert result["status"] == "needs_review"
    assert result["attempts"] == 2
    assert result["draft"] is None
