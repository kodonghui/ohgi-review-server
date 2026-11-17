"""강의 규정 Markdown을 조문별로 분할해 Chroma 인덱스를 만든다.

서버를 켜기 전에 실행한다. 새 인덱스는 완성 후 manifest로 활성화한다.
"""

import argparse
import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import httpx
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pydantic import SecretStr

from app.core.config import PROJECT_ROOT, get_settings
from app.services.hybrid_retriever import HybridRetriever


REGULATIONS = PROJECT_ROOT / "data" / "ohgi_regulations.md"
COLLECTION = "ohgi_regulations"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072
INDEX_VERSION = 1


@dataclass(frozen=True)
class IndexEmbedding:
    model: str
    function: Embeddings
    dimensions: int


def openai_policy_embedding(api_key: SecretStr | None, *, http_client: httpx.Client | None = None) -> IndexEmbedding:
    if not api_key or not api_key.get_secret_value().strip():
        raise ValueError("정책 임베딩에 OPENAI_API_KEY가 필요합니다.")
    function = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        http_client=http_client,
        request_timeout=60,
        max_retries=1,
    )
    return IndexEmbedding(EMBEDDING_MODEL, function, EMBEDDING_DIMENSIONS)


def load_articles(source: Path = REGULATIONS) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter([("##", "chapter"), ("###", "article")])
    sections = splitter.split_text(source.read_text(encoding="utf-8"))
    return [
        Document(
            id=section.metadata["article"].split()[0],
            page_content=f"{section.metadata['article']}\n{section.page_content}",
            metadata={
                **section.metadata,
                "source": source.name,
                "article_no": section.metadata["article"].split()[0],
            },
        )
        for section in sections
        if "article" in section.metadata
    ]


def open_policy_index(runtime_dir: Path, *, embedding: IndexEmbedding | None = None) -> Chroma | None:
    manifest_path = runtime_dir / "policy-index.json"
    if not manifest_path.is_file():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_name = embedding.model if embedding else EMBEDDING_MODEL
    dimensions = embedding.dimensions if embedding else EMBEDDING_DIMENSIONS
    if (
        manifest["source_sha256"] != hashlib.sha256(REGULATIONS.read_bytes()).hexdigest()
        or manifest["embedding_model"] != model_name
        or manifest.get("embedding_dimensions") != dimensions
        or manifest["version"] != INDEX_VERSION
    ):
        raise ValueError("규정 또는 임베딩 설정이 바뀌었습니다. 정책 인덱스를 다시 만드세요.")
    directory = runtime_dir / "policy-indexes" / manifest["directory"]
    if not (directory / "chroma.sqlite3").is_file():
        raise FileNotFoundError("정책 인덱스 파일을 찾을 수 없습니다.")
    if embedding is None:
        embedding = openai_policy_embedding(get_settings().openai_api_key)
    index = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding.function,
        persist_directory=str(directory),
        client_settings=ChromaSettings(anonymized_telemetry=False),
        create_collection_if_not_exists=False,
    )
    if len(index.get(include=[])["ids"]) != manifest["article_count"]:
        raise ValueError("정책 인덱스에 저장된 조문 수가 manifest와 다릅니다.")
    return index


def build_policy_index(runtime_dir: Path, *, embedding: IndexEmbedding | None = None) -> dict:
    if embedding is None:
        embedding = openai_policy_embedding(get_settings().openai_api_key)
    source_hash = hashlib.sha256(REGULATIONS.read_bytes()).hexdigest()
    directory_name = uuid4().hex
    directory = runtime_dir / "policy-indexes" / directory_name
    directory.mkdir(parents=True)
    articles = load_articles()
    index = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding.function,
        persist_directory=str(directory),
        client_settings=ChromaSettings(anonymized_telemetry=False),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    ids = [article.metadata["article_no"] for article in articles]
    index.add_documents(articles, ids=ids)
    if set(index.get(include=[])["ids"]) != set(ids):
        raise RuntimeError("인덱스에 모든 조문이 저장되지 않았습니다.")

    manifest = {
        "directory": directory_name,
        "version": INDEX_VERSION,
        "source_sha256": source_hash,
        "embedding_model": embedding.model,
        "embedding_dimensions": embedding.dimensions,
        "article_count": len(ids),
    }
    # 저장 중 프로세스가 끝나도 활성 manifest는 직전 완성 인덱스를 가리킨다.
    temporary = runtime_dir / f"policy-index-{directory_name}.json"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(runtime_dir / "policy-index.json")
    return manifest


async def evaluate_retrieval(index: Chroma) -> dict:
    retriever = HybridRetriever(index, load_articles())
    samples = json.loads((REGULATIONS.parent / "golden_set.json").read_text(encoding="utf-8"))
    rows = []
    summary = {name: {"hit_at_1": 0, "hit_at_3": 0} for name in ("vector", "bm25", "hybrid")}
    for sample in samples:
        ranked = await retriever.search(sample["question"])
        top = {name: [doc.id for doc in getattr(ranked, name)[:3]] for name in summary}
        for name, ids in top.items():
            summary[name]["hit_at_1"] += sample["source"] in ids[:1]
            summary[name]["hit_at_3"] += sample["source"] in ids
        rows.append({"question": sample["question"], "expected": sample["source"], **top})
    return {"questions": len(rows), "summary": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true", help="기존 인덱스로 강의 골든셋 Hit@1·Hit@3를 확인합니다.")
    args = parser.parse_args()
    runtime_dir = get_settings().runtime_dir
    if not args.evaluate:
        print(json.dumps(build_policy_index(runtime_dir), ensure_ascii=False))
        return

    index = open_policy_index(runtime_dir)
    if index is None:
        parser.error("먼저 정책 인덱스를 만드세요.")
    result = asyncio.run(evaluate_retrieval(index))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
