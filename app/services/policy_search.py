"""질문별 규정 원문을 검색하고, 근거가 없는 답글을 보류한다."""

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from chromadb.errors import ChromaError
from langchain_core.documents import Document

from app.services.hybrid_retriever import HybridRetriever, PolicyVectorIndex, SearchRankings
from app.services.policy_passages import (
    ACTION_TERMS, CONTEXT_QUESTION, DOCUMENT_TOPIC, FOCUS_TERMS, PolicyPassages, normalized, split_questions, terms,
)
from app.services.policy_topics import (
    POLICY_QUESTION, QUESTION_FIELD, PolicyEvidence, PolicyRule, article_sentences, question_clauses, route_questions,
)


FIRST_PASS_ARTICLES = 3
MAX_CONTEXT_ARTICLES = 6
MAX_DOCUMENT_QUESTIONS = 6
SEARCH_ERRORS = (ChromaError, OSError, RuntimeError)
ARTICLE_REFERENCE = re.compile(r"제\s*(\d+)\s*조")


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


def find_evidence(rule: PolicyRule, candidates: list[Document]) -> tuple[Document, PolicyEvidence] | None:
    for article in candidates:
        if article.id != rule.article_id or not valid_article(article):
            continue
        evidence = rule.evidence(article)
        if evidence is not None:
            return article, evidence
    return None


class PolicySearch:
    def __init__(
        self, index: PolicyVectorIndex | None, hybrid: HybridRetriever | None = None, articles: Sequence[Document] = (),
    ):
        self._index = index
        self._hybrid = hybrid
        self._passages = PolicyPassages(articles) if articles else None

    async def retrieve(self, review_text: str) -> PolicySearchResult:
        questions, unresolved, reason = route_questions(review_text)
        if reason or not (questions or unresolved):
            return PolicySearchResult((), (), bool(reason), reason=reason)

        required_ids = tuple(dict.fromkeys(rule.article_id for rule, _ in questions))
        if len(required_ids) > MAX_CONTEXT_ARTICLES:
            return PolicySearchResult(required_ids, (), True, reason="context_limit")
        if unresolved and (self._hybrid is None or self._passages is None):
            return PolicySearchResult(required_ids, (), True, reason="unknown_question")
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
        open_questions = dict.fromkeys(unresolved)
        failed = False

        async def search(query: str) -> SearchRankings:
            assert self._hybrid is not None
            if query not in rankings:
                rankings[query] = await self._hybrid.search(query)
            return rankings[query]

        def accept(article: Document, item: PolicyEvidence, question: str) -> None:
            articles[article.id] = article
            evidence.append(replace(item, question=question))

        try:
            for rule, question in questions:
                if rule.direct:
                    article = by_id.get(rule.article_id)
                    found = find_evidence(rule, [article]) if article is not None else None
                else:
                    found = None
                    if self._hybrid is not None:
                        initial = await search(question)
                        found = find_evidence(rule, initial.hybrid[:FIRST_PASS_ARTICLES])
                        if found is None:
                            retried.append(rule.topic)
                            focused = await search(rule.query)
                            # RRF가 먼저 제외한 후보까지 복원한 뒤 해당 문의에 답하는 원문을 확인한다.
                            found = find_evidence(rule, focused.hybrid + initial.hybrid)
                if found is None:
                    if not rule.direct and self._passages:
                        open_questions[question] = None
                    else:
                        failed = True
                    continue
                article, item = found
                if not rule.direct and self._passages:
                    parts = split_questions(question)
                    missing_parts = [
                        part for part in parts
                        if not self._passages.matches_scope(part, item, check_form=len(parts) > 1)
                    ]
                    open_questions.update(dict.fromkeys(missing_parts))
                    if len(missing_parts) == len(parts):
                        continue
                accept(article, item, question)
            if open_questions and self._hybrid and self._passages:
                contexts = question_contexts(review_text)
                parts = {part: contexts.get(text, "") for text in open_questions for part in split_questions(text)}
                if len(parts) > MAX_DOCUMENT_QUESTIONS:
                    return PolicySearchResult(tuple(articles), tuple(articles.values()), True, tuple(evidence),
                                              "context_limit", tuple(retried))
                full_query = review_text.strip()
                unanswered = {}
                for question, context in parts.items():
                    # Keep declarations for a single question, without mixing two different requests.
                    query = full_query if len(contexts) == 1 else question
                    candidates = (await search(query)).hybrid
                    found = self._passages.select(question, candidates, context=context)
                    if found is None:
                        search_query = " ".join(normalized(f"{question} {context}").split())
                        if search_query and search_query not in rankings:
                            retried.append(DOCUMENT_TOPIC)
                            await search(search_query)
                        if search_query in rankings:
                            candidates = candidate_union(rankings[search_query].hybrid, candidates)
                        found = self._passages.select(question, candidates, context=context)
                    if found is None:
                        unanswered[question] = (context, candidates)
                        continue
                    accept(*found, question)

                if unanswered:
                    references = {}
                    for question, (context, candidates) in unanswered.items():
                        query = self._passages.query_terms(question, context)
                        references[question] = reference_candidates(query, candidate_union(candidates, list(articles.values())))
                    reference_ids = list(dict.fromkeys(
                        article_id for ids in references.values() for article_id in ids
                    ))[:MAX_CONTEXT_ARTICLES]
                    if reference_ids:
                        linked = {article.id: article for article in await self._index.aget_by_ids(reference_ids)
                                  if valid_article(article)}
                        if linked:
                            retried.append("article_reference")
                            for question, (context, _) in tuple(unanswered.items()):
                                candidates = [linked[article_id] for article_id in references[question]
                                              if article_id in linked]
                                if found := self._passages.select(question, candidates, context=context):
                                    accept(*found, question)
                                    del unanswered[question]
                failed |= bool(unanswered)
        except SEARCH_ERRORS:
            return PolicySearchResult(required_ids, tuple(articles.values()), True, tuple(evidence),
                                      "index_unavailable", tuple(retried))

        resolved_ids = tuple(articles)
        if len(resolved_ids) > MAX_CONTEXT_ARTICLES:
            return PolicySearchResult(resolved_ids, tuple(articles.values()), True, tuple(evidence),
                                      "context_limit", tuple(retried))
        required = tuple(dict.fromkeys((*required_ids, *resolved_ids))) if failed else resolved_ids
        return PolicySearchResult(required, tuple(articles.values()), failed, tuple(evidence),
                                  "evidence_missing" if failed else "", tuple(retried))


def question_contexts(text: str) -> dict[str, str]:
    """Keep declarations and the immediately preceding topic for an elliptical follow-up."""
    declarations = []
    contexts = {}
    previous = ""
    for part in question_clauses(text):
        if POLICY_QUESTION.search(part) or QUESTION_FIELD.search(part):
            nearby = [*declarations, previous] if CONTEXT_QUESTION.search(part) else declarations
            contexts[part] = " ".join(fragment for fragment in nearby if fragment)
            previous = part
        else:
            declarations.append(part)
    return contexts


def reference_candidates(query: frozenset[str], candidates: list[Document]) -> tuple[str, ...]:
    """Expand one hop from sentences that point to a rule for this question's scope."""
    focus = (query & FOCUS_TERMS) - ACTION_TERMS
    references = []
    for article in candidates:
        for sentence in article_sentences(article):
            numbers = ARTICLE_REFERENCE.findall(sentence)
            if not numbers:
                continue
            matching_terms = query & terms(normalized(sentence))
            if (focus and focus <= matching_terms) or (not focus and len(matching_terms) >= 2):
                references.extend(f"제{number}조" for number in numbers if f"제{number}조" != article.id)
    return tuple(dict.fromkeys(references))


def candidate_union(*rankings: list[Document]) -> list[Document]:
    by_id = {}
    for article in (article for rank in rankings for article in rank):
        if valid_article(article):
            by_id.setdefault(article.id, article)
    return list(by_id.values())
