"""필수 조문을 ID로 확보하고, Chroma에서 관련 규정 후보를 검색한다."""

import re
from dataclasses import dataclass

from chromadb.errors import ChromaError
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.services.hybrid_retriever import HybridRetriever

QUALITY_ARTICLE = "제12조"
DELIVERY_ARTICLE = "제17조"

# 두 조문이 함께 적용될 수 있으므로 각각 확인한다.
QUALITY_ISSUE = re.compile(r"이물질|오배송|다른\s*메뉴|다른\s*음식|메뉴가\s*잘못|주문한\s*것과\s*다른")
TASTE_COMPLAINT = re.compile(r"(?:맛|입맛)[^.!?\n]{0,24}(?:안\s*맞|맞지\s*않|별로|없)")
DELIVERY_DAMAGE = re.compile(
    r"(?:용기|포장|배달)[^.!?\n]{0,16}(?:파손|깨졌|깨져|터졌)|쏟아졌|쏟아져"
)
POLICY_REQUEST = re.compile(r"환불|보상|재조리|재배달|교환")
POLICY_INFORMATION = re.compile(
    r"영업\s*시간|쉬는\s*날|휴무|주차|브레이크\s*타임|배달비|포장\s*가능|예약\s*가능|"
    r"마지막\s*주문|공휴일|언제\s*(?:열|닫|마감)|(?:얼마|몇\s*시)[^.!?\n]{0,14}(?:인가|예요|까지)"
)
UNSUPPORTED_POLICY = re.compile(
    r"알레르기|쿠폰[^.!?\n]{0,18}(?:발급|보상|환불|취소|주세요|주실|줘|달라|준다고|드린다고|안\s*됐)"
)


@dataclass(frozen=True)
class PolicySearchResult:
    required_ids: tuple[str, ...]
    articles: tuple[Document, ...]
    needs_manual_review: bool
    required_hours: tuple[str, ...] = ()


class PolicySearch:
    def __init__(self, index: Chroma | None, hybrid: HybridRetriever | None = None):
        self._index = index
        self._hybrid = hybrid

    async def retrieve(self, review_text: str) -> PolicySearchResult:
        if UNSUPPORTED_POLICY.search(review_text):
            return PolicySearchResult((), (), True)

        quality_incident = bool(QUALITY_ISSUE.search(review_text))
        delivery_incident = bool(DELIVERY_DAMAGE.search(review_text))
        quality_issue = quality_incident or (
            "환불" in review_text and TASTE_COMPLAINT.search(review_text)
        )
        required_ids = tuple(
            article_id
            for applies, article_id in (
                (quality_issue, QUALITY_ARTICLE),
                (delivery_incident, DELIVERY_ARTICLE),
            )
            if applies
        )
        if not required_ids and POLICY_REQUEST.search(review_text):
            return PolicySearchResult((), (), True)
        asks_for_information = bool(POLICY_INFORMATION.search(review_text))
        if not required_ids and not asks_for_information:
            return PolicySearchResult((), (), False)
        if self._index is None:
            return PolicySearchResult(required_ids, (), True)

        try:
            matches = await self._index.aget_by_ids(required_ids) if required_ids else []
        except (ChromaError, OSError, RuntimeError):
            return PolicySearchResult(required_ids, (), True)
        by_id = {
            item.id: item
            for item in matches
            if item.id == item.metadata.get("article_no") and item.page_content.strip()
        }
        required = tuple(by_id[article_id] for article_id in required_ids if article_id in by_id)
        if len(required) != len(required_ids):
            return PolicySearchResult(required_ids, required, True)
        incidents = {
            article_id
            for occurred, article_id in ((quality_incident, QUALITY_ARTICLE), (delivery_incident, DELIVERY_ARTICLE))
            if occurred
        }
        # 접수 기한은 현재 조문에서 가져온다. 맛 불만만 있는 경우 제12조의 예외 기한을 강제하지 않는다.
        required_hours = tuple(
            hours
            for article in required
            if article.id in incidents
            for hours in re.findall(r"(\d+)\s*시간", article.page_content)[:1]
        )
        if self._hybrid is None:
            return PolicySearchResult(required_ids, required, not required, required_hours)

        try:
            rankings = await self._hybrid.search(review_text)
        except (ChromaError, OSError, RuntimeError):
            return PolicySearchResult(required_ids, required, not required, required_hours)
        extra = tuple(
            article for article in rankings.hybrid
            if article.id and article.id not in by_id
            and article.id == article.metadata.get("article_no") and article.page_content.strip()
        )
        articles = (required + extra)[:3]
        return PolicySearchResult(required_ids, articles, not articles, required_hours)
