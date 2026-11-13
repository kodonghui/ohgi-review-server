"""Chroma 벡터 검색과 BM25의 후보를 순위로 합친다."""

import re
from collections import defaultdict
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


CANDIDATE_COUNT = 8
RRF_K = 60


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[가-힣a-z0-9]+", text.lower())
    bigrams = [
        f"2:{word[index:index + 2]}"
        for word in words
        if re.search(r"[가-힣]", word)
        for index in range(len(word) - 1)
    ]
    return words + bigrams


def reciprocal_rank_fusion(*rankings: list[Document]) -> list[Document]:
    scores: dict[str, float] = defaultdict(float)
    documents: dict[str, Document] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, document in enumerate(ranking, start=1):
            article_id = document.metadata["article_no"]
            if article_id not in seen:
                seen.add(article_id)
                documents[article_id] = document
                scores[article_id] += 1 / (RRF_K + rank)
    ids = sorted(scores, key=lambda article_id: (-scores[article_id], article_id))
    return [documents[article_id] for article_id in ids]


@dataclass(frozen=True)
class SearchRankings:
    vector: list[Document]
    bm25: list[Document]
    hybrid: list[Document]


class HybridRetriever:
    def __init__(self, index: Chroma, articles: list[Document]):
        self._index = index
        self._articles = articles
        self._bm25 = BM25Okapi([tokenize(article.page_content) for article in articles])

    async def search(self, query: str) -> SearchRankings:
        vector = await self._index.asimilarity_search(query, k=CANDIDATE_COUNT)
        scores = self._bm25.get_scores(list(dict.fromkeys(tokenize(query))))
        ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        bm25 = [self._articles[index] for index in ranked[:CANDIDATE_COUNT] if scores[index] > 0]
        return SearchRankings(vector, bm25, reciprocal_rank_fusion(vector, bm25))
