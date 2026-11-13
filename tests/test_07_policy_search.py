"""정책 문의는 모두 근거를 얻거나 초안 작성 전에 보류된다."""

import asyncio

from langchain_core.documents import Document

from app.services.hybrid_retriever import SearchRankings
from app.services.policy_index import load_articles
from app.services.policy_search import PolicySearch
from app.services.policy_topics import RULES


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


def test_order_status_stops_even_if_general_damage_policy_is_known():
    index = TestIndex()
    hybrid = TestRetriever({})
    query = "배달 용기가 깨졌는데 제 주문 환불이 이미 승인됐는지 알려 주세요."
    result = asyncio.run(PolicySearch(index, hybrid).retrieve(query))
    assert result.needs_manual_review is True
    assert result.reason == "staff_only"
    assert hybrid.queries == []
    assert index.lookups == []


def test_question_outside_the_policy_cannot_continue_as_regular_praise():
    index = TestIndex()
    result = asyncio.run(PolicySearch(index).retrieve("토요일에 케이크를 가져가도 되나요?"))
    assert result.needs_manual_review is True
    assert result.reason == "unknown_question"
