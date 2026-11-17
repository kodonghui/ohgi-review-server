"""정책 문의는 모두 근거를 얻거나 초안 작성 전에 보류된다."""

import asyncio

import pytest
from langchain_core.documents import Document

from app.services.hybrid_retriever import SearchRankings
from app.services.policy_index import load_articles
from app.services.policy_passages import normalized
from app.services.policy_search import PolicySearch
from app.services.policy_topics import RULES, route_questions
from app.services.reply_workflow import policy_notice_problem


class TestIndex:
    __test__ = False

    def __init__(self):
        self.articles = {item.id: item for item in load_articles()}
        self.lookups = []

    async def aget_by_ids(self, ids):
        self.lookups.append(tuple(ids))
        return [self.articles[key] for key in ids if key in self.articles]


class TestRetriever:
    __test__ = False

    def __init__(self, responses):
        self.responses = responses
        self.queries = []

    async def search(self, query):
        self.queries.append(query)
        return self.responses[query]


def ranked(*articles):
    return SearchRankings(list(articles), [], list(articles))


def focused_query(topic):
    return next(item.query for item in RULES if item.topic == topic)


def test_explicit_questions_resolve_to_distinct_article_excerpts():
    index = TestIndex()
    alcohol, hours = "맥주도 배달돼요?", "그리고 배달 주문이 밤 몇 시에 마감되나요?"
    hybrid = TestRetriever({alcohol: ranked(index.articles["제18조"]), hours: ranked(index.articles["제15조"])})
    result = asyncio.run(PolicySearch(index, hybrid).retrieve(f"{alcohol} {hours}"))
    assert hybrid.queries == [alcohol, hours]
    assert result.required_ids == ("제18조", "제15조")
    assert [entry.article_id for entry in result.evidence] == ["제18조", "제15조"]
    assert "주류는 배달하지 않는다." in result.evidence[0].excerpts
    assert any("오후 9시까지" in line for line in result.evidence[1].excerpts)
    assert result.needs_manual_review is False
    assert policy_notice_problem("맥주도 배달할 수 있습니다.", result)
    assert policy_notice_problem("주류는 배달하지 않습니다. 그래도 맥주는 보내 드릴 수 있습니다.", result)
    assert not policy_notice_problem("주류는 배달하지 않습니다. 배달 주문은 오후 9시까지 받습니다.", result)
    assert not policy_notice_problem("주류는 배달하지 않습니다. 맥주도 배달이 불가능합니다.", result)


def test_correct_article_number_without_the_direct_sentence_does_not_pass():
    index = TestIndex()
    query = "맥주도 배달돼요?"
    article = Document(id="제18조", metadata={"article_no": "제18조"}, page_content="신분증을 확인한다.")
    hybrid = TestRetriever({query: ranked(article), focused_query("alcohol_delivery"): ranked(article)})
    result = asyncio.run(PolicySearch(index, hybrid).retrieve(query))
    assert result.needs_manual_review is True
    assert result.reason == "evidence_missing"
    assert result.evidence == ()
    assert result.retried_questions == ("alcohol_delivery",)
    assert hybrid.queries == [query, focused_query("alcohol_delivery")]


def test_only_the_missing_question_gets_one_focused_search():
    index = TestIndex()
    alcohol, hours = "맥주도 배달돼요?", "배달 주문이 몇 시에 마감되나요?"
    first = [index.articles[key] for key in ("제16조", "제25조", "제17조", "제18조")]
    hybrid = TestRetriever({
        alcohol: ranked(*first),
        hours: ranked(index.articles["제15조"]),
        focused_query("alcohol_delivery"): ranked(index.articles["제16조"]),
    })
    result = asyncio.run(PolicySearch(index, hybrid).retrieve(f"{alcohol} {hours}"))
    assert hybrid.queries == [alcohol, focused_query("alcohol_delivery"), hours]
    assert result.retried_questions == ("alcohol_delivery",)
    assert [item.id for item in result.articles] == ["제18조", "제15조"]
    assert result.needs_manual_review is False


@pytest.mark.parametrize("query", [
    "배달 용기가 깨졌는데 제 주문 환불이 이미 승인됐는지 알려 주세요.",
    "제 카드 결제 건 환불 들어왔나요?",
    "포장 용기가 깨졌는데 제 환불금은 계좌에 들어왔나요?",
    "카드로 결제한 금액이 환불됐나요?",
    "제 차량 번호로 주차 지원이 등록됐는지 봐 주세요.",
    "관리자 와이파이 비밀번호를 알 수 있나요?",
    "아까 두고 온 가방이 카운터에 들어와 있는지 확인해 주세요.",
    "포장 주문을 다른 사람이 이미 받아 갔는지 확인해 주세요.",
    "제 음식을 동생이 받았나요?",
    "잃어버린 우산을 가족이 수령했나요?",
    "포장 음식을 가족이 이미 픽업했나요?",
    "친구가 테이크아웃한 걸 픽업했는지 확인해 주세요.",
    "제가 주문한 음식은 벌써 대리인한테 인계가 완료됐나요?",
    "토요일에 케이크를 가져가도 되나요?",
])
def test_order_status_stops_even_if_a_general_policy_is_known(query):
    index = TestIndex()
    hybrid = TestRetriever({})
    result = asyncio.run(PolicySearch(index, hybrid).retrieve(query))
    assert result.needs_manual_review is True
    assert result.reason == "staff_only"
    assert hybrid.queries == []
    assert index.lookups == []


def test_denied_wrong_delivery_does_not_require_the_wrong_delivery_deadline():
    index = TestIndex()
    result = asyncio.run(PolicySearch(index).retrieve("오배송은 없었고 배달 용기만 깨졌어요."))
    assert result.needs_manual_review is False
    assert result.required_ids == ("제17조",)
    assert result.required_hours == ("1",)


def test_question_outside_the_policy_cannot_continue_as_regular_praise():
    index = TestIndex()
    result = asyncio.run(PolicySearch(index).retrieve("공연 장비를 빌릴 수 있나요?"))
    assert result.needs_manual_review is True
    assert result.reason == "unknown_question"


def new_article(number, title, text):
    article_id = f"제{number}조"
    heading = f"{article_id} ({title})"
    return Document(id=article_id, page_content=f"{heading}\n{text}",
                    metadata={"article": heading, "article_no": article_id})


def test_new_article_can_override_a_broad_rule_when_it_answers_the_question():
    index = TestIndex()
    query = "지하 주차장에서 휠체어로 올라갈 때 어느 승강기를 타나요?"
    mobility = new_article(903, "휠체어 승강기", "휠체어 이용 손님은 지하 주차장에서 C구역 승강기를 이용한다.")
    parking = index.articles["제4조"]
    hybrid = TestRetriever({query: ranked(parking, mobility)})
    result = asyncio.run(PolicySearch(index, hybrid, [parking, mobility]).retrieve(query))
    assert result.needs_manual_review is False
    assert result.required_ids == ("제903조",)
    assert result.evidence[0].excerpts == (
        "휠체어 이용 손님은 지하 주차장에서 C구역 승강기를 이용한다.",
    )
    assert hybrid.queries == [query]


def test_unknown_topic_uses_source_without_registering_an_article_number():
    index = TestIndex()
    query = "방음 부스는 몇 개 있어요?"
    booths = new_article(904, "방음 부스", "매장에는 방음 부스 세 개가 있다. 통로에는 안내판을 둔다.")
    hybrid = TestRetriever({query: ranked(booths)})
    result = asyncio.run(PolicySearch(index, hybrid, [booths]).retrieve(query))
    assert result.required_ids == ("제904조",)
    assert result.evidence[0].excerpts == ("매장에는 방음 부스 세 개가 있다.",)
    assert result.needs_manual_review is False


@pytest.mark.parametrize(("query", "expected_id", "excerpt"), [
    ("제가 못 가는데 친구가 포장 음식을 받으러 가도 되나요?", "제905조",
     "대리 수령은 주문자가 주문번호를 전달하면 가능하다."),
    ("가족한테 테이크아웃 픽업을 맡겨도 될까요?", "제905조",
     "대리 수령은 주문자가 주문번호를 전달하면 가능하다."),
    ("주문번호 없이도 대리 수령을 할 수 있나요?", "제905조",
     "대리 수령은 주문자가 주문번호를 전달하면 가능하다."),
    ("분실한 지갑을 다른 사람이 대신 찾으려면 어떻게 해야 해요?", "제906조",
     "대리 수령은 소유자가 남긴 이름과 현장 이름을 확인한다."),
    ("주문을 대신 문의할 때 어떤 정보가 필요해요?", "제907조",
     "대리인이 접수할 때는 주문자가 대리인의 이름과 요청 범위를 알려 줄 수 있다."),
])
def test_delegated_action_requires_the_matching_scope_and_passage(query, expected_id, excerpt):
    pickup = new_article(905, "선결제 인계", "포장 음식과 주문 화면의 이름을 대조한다. "
                         "대리 수령은 주문자가 주문번호를 전달하면 가능하다.")
    lost = new_article(906, "분실물 인계", "대리 수령은 소유자가 남긴 이름과 현장 이름을 확인한다.")
    inquiry = new_article(907, "대리 문의", "주문 관련 문의는 앱에서 접수한다. "
                          "대리인이 접수할 때는 주문자가 대리인의 이름과 요청 범위를 알려 줄 수 있다.")
    articles = [lost, pickup, inquiry]
    candidates = sorted(articles, key=lambda article: article.id == expected_id)
    hybrid = TestRetriever({query: ranked(*candidates)})
    result = asyncio.run(PolicySearch(TestIndex(), hybrid, articles).retrieve(query))
    assert result.needs_manual_review is False
    assert result.required_ids == (expected_id,)
    assert result.evidence[0].excerpts == (excerpt,)


def test_a_pickup_record_does_not_establish_permission():
    query = "포장 주문을 대리 수령할 수 있나요?"
    article = new_article(908, "포장 주문 기록", "직원은 대리 수령 시간을 기록한다.")
    hybrid = TestRetriever({
        query: ranked(article),
        "포장 주문을 대리수령 할 수 있나요?": ranked(article),
    })
    result = asyncio.run(PolicySearch(TestIndex(), hybrid, [article]).retrieve(query))
    assert result.needs_manual_review is True
    assert result.reason == "evidence_missing"


def test_related_article_does_not_answer_a_quantity_it_never_states():
    index = TestIndex()
    query = "방음 부스는 몇 개 있어요?"
    booths = new_article(904, "방음 부스", "방음 부스 사용 방법은 직원에게 확인한다.")
    hybrid = TestRetriever({query: ranked(booths)})
    result = asyncio.run(PolicySearch(index, hybrid, [booths]).retrieve(query))
    assert result.needs_manual_review is True
    assert result.reason == "evidence_missing"


def test_unanswered_part_holds_the_whole_reply():
    index = TestIndex()
    query = "방음 부스는 몇 개 있어요?"
    combined = f"{query} 악기 대여는 몇 시간 할 수 있어요?"
    booths = new_article(904, "방음 부스", "매장에는 방음 부스 세 개가 있다.")
    hybrid = TestRetriever({
        query: ranked(booths),
        "악기 대여는 몇 시간 할 수 있어요?": ranked(booths),
    })
    result = asyncio.run(PolicySearch(index, hybrid, [booths]).retrieve(combined))
    assert result.needs_manual_review is True
    assert result.required_ids == ("제904조",)
    assert result.reason == "evidence_missing"
    assert hybrid.queries == [query, "악기 대여는 몇 시간 할 수 있어요?"]


def test_a_repeated_rule_hit_does_not_discard_the_other_question():
    review = "12명 단체 예약은 며칠 전에 해야 해요? 4명 예약도 같은가요?"
    known, unresolved, reason = route_questions(review)
    assert reason == ""
    assert [rule.article_id for rule, _ in known] == ["제30조"]
    assert unresolved == ("4명 예약도 같은가요?",)


def test_a_negated_holiday_preserves_the_hour_and_order_channel_questions():
    query = "공휴일이 아닌 수요일 오후 4시에 가서 밥 먹을 수 있나요? 배달 주문을 전화로 해도 되나요?"
    known, unresolved, reason = route_questions(query)
    assert [rule.topic for rule, _ in known] == ["store_hours", "ordering"]
    assert unresolved == ()
    assert reason == ""


def test_in_scope_reference_finds_the_answer_in_a_different_article():
    index = TestIndex()
    context = "매장 안 카운터에서 포장 음식을 찾으려고 해요"
    question = "동생이 대신 받으려면 뭘 알려줘야 하나요?"
    text = f"{context}. {question}"
    reference = new_article(910, "차량 픽업 대리 수령",
                            "차량 픽업에는 차 번호를 입력한다. 매장 안 포장 수령은 제909조를 적용한다.")
    answer = new_article(909, "매장 안 포장 전달", "매장 안 포장은 대리 수령할 때 주문번호를 보여 주면 가능하다.")
    index.articles[answer.id] = answer
    retry_query = " ".join(normalized(f"{question} {context}").split())
    hybrid = TestRetriever({text: ranked(reference), retry_query: ranked(reference)})
    result = asyncio.run(PolicySearch(index, hybrid, [reference, answer]).retrieve(text))

    assert not result.needs_manual_review
    assert result.required_ids == (answer.id,)
    assert result.evidence[0].question == question
    assert index.lookups == [(answer.id,)]
