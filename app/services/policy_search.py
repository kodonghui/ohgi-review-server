"""질문별 규정 원문을 검색하고, 근거가 없는 답글을 보류한다."""

from dataclasses import dataclass

from chromadb.errors import ChromaError
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.services.hybrid_retriever import HybridRetriever, SearchRankings
from app.services.policy_topics import PolicyEvidence, PolicyRule, questions_in


MAX_CONTEXT_ARTICLES = 3
SEARCH_ERRORS = (ChromaError, OSError, RuntimeError)


@dataclass(frozen=True)
class PolicySearchResult:
    required_ids: tuple[str, ...]
    articles: tuple[Document, ...]
    needs_manual_review: bool
    evidence: tuple[PolicyEvidence, ...] = ()
    reason: str = ""
    retried_questions: tuple[str, ...] = ()

    @property
    def required_hours(self) -> tuple[str, ...]:
        return tuple(item.hours for item in self.evidence if item.hours is not None)

    @property
    def requires_followup(self) -> bool:
        return any(item.followup for item in self.evidence)


def valid_article(article: Document) -> bool:
    return bool(article.id and article.id == article.metadata.get("article_no") and article.page_content.strip())


def find_evidence(
    rule: PolicyRule, question: str, candidates: list[Document],
) -> tuple[Document, PolicyEvidence] | None:
    for article in candidates:
        if article.id != rule.article_id or not valid_article(article):
            continue
        evidence = rule.evidence(article, question)
        if evidence is not None:
            return article, evidence
    return None


class PolicySearch:
    def __init__(self, index: Chroma | None, hybrid: HybridRetriever | None = None):
        self._index = index
        self._hybrid = hybrid

    async def retrieve(self, review_text: str) -> PolicySearchResult:
        questions, reason = questions_in(review_text)
        if reason or not questions:
            return PolicySearchResult((), (), bool(reason), reason=reason)

        required_ids = tuple(dict.fromkeys(rule.article_id for rule, _ in questions))
        if len(required_ids) > MAX_CONTEXT_ARTICLES:
            return PolicySearchResult(required_ids, (), True, reason="context_limit")
        if self._index is None:
            return PolicySearchResult(required_ids, (), True, reason="index_unavailable")

        direct_ids = tuple(dict.fromkeys(rule.article_id for rule, _ in questions if rule.direct))
        by_id = {}
        if direct_ids:
            try:
                by_id = {article.id: article for article in await self._index.aget_by_ids(direct_ids)
                         if valid_article(article)}
            except SEARCH_ERRORS:
                return PolicySearchResult(required_ids, (), True, reason="index_unavailable")

        rankings: dict[str, SearchRankings] = {}
        articles: dict[str, Document] = {}
        evidence: list[PolicyEvidence] = []
        retried = []
        try:
            for rule, question in questions:
                if rule.direct:
                    article = by_id.get(rule.article_id)
                    found = find_evidence(rule, question, [article]) if article is not None else None
                else:
                    found = None
                    if self._hybrid is not None:
                        if question not in rankings:
                            rankings[question] = await self._hybrid.search(question)
                        initial = rankings[question]
                        found = find_evidence(rule, question, initial.hybrid[:MAX_CONTEXT_ARTICLES])
                        if found is None:
                            retried.append(rule.topic)
                            focused = await self._hybrid.search(rule.query)
                            # RRF가 먼저 제외한 후보까지 복원한 뒤 해당 문의에 답하는 원문을 확인한다.
                            found = find_evidence(rule, question, focused.hybrid + initial.hybrid)
                if found is not None:
                    article, item = found
                    articles[rule.article_id] = article
                    evidence.append(item)
        except SEARCH_ERRORS:
            return PolicySearchResult(required_ids, tuple(articles.values()), True, tuple(evidence),
                                      "index_unavailable", tuple(retried))

        missing = len(evidence) < len(questions)
        return PolicySearchResult(required_ids, tuple(articles.values()), missing, tuple(evidence),
                                  "evidence_missing" if missing else "", tuple(retried))
