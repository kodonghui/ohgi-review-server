"""질문별 규정 원문을 검색하고, 근거가 없는 답글을 보류한다."""

from dataclasses import dataclass

from chromadb.errors import ChromaError
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.services.hybrid_retriever import HybridRetriever
from app.services.policy_topics import PolicyEvidence, questions_in


MAX_CONTEXT_ARTICLES = 3
SEARCH_ERRORS = (ChromaError, OSError, RuntimeError)


@dataclass(frozen=True)
class PolicySearchResult:
    required_ids: tuple[str, ...]
    articles: tuple[Document, ...]
    needs_manual_review: bool
    evidence: tuple[PolicyEvidence, ...] = ()
    reason: str = ""

    @property
    def required_hours(self) -> tuple[str, ...]:
        return tuple(item.hours for item in self.evidence if item.hours is not None)

    @property
    def requires_followup(self) -> bool:
        return any(item.followup for item in self.evidence)


def valid_article(article: Document) -> bool:
    return bool(article.id and article.id == article.metadata.get("article_no") and article.page_content.strip())


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

        articles: dict[str, Document] = {}
        evidence = []
        missing = False
        for rule, question in questions:
            article = by_id.get(rule.article_id) if rule.direct else None
            if not rule.direct and self._hybrid is not None:
                try:
                    candidates = (await self._hybrid.search(question)).hybrid[:MAX_CONTEXT_ARTICLES]
                except SEARCH_ERRORS:
                    return PolicySearchResult(required_ids, tuple(articles.values()), True, tuple(evidence),
                                              "index_unavailable")
                article = next((item for item in candidates if item.id == rule.article_id and valid_article(item)), None)
            item = rule.evidence(article, question) if article is not None else None
            if item is None:
                missing = True
                continue
            articles[rule.article_id] = article
            evidence.append(item)
        return PolicySearchResult(required_ids, tuple(articles.values()), missing, tuple(evidence),
                                  "evidence_missing" if missing else "")
