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

from chromadb.config import Settings as ChromaSettings
from fastembed import TextEmbedding
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.core.config import PROJECT_ROOT, get_settings
from app.services.hybrid_retriever import HybridRetriever


REGULATIONS = PROJECT_ROOT / "data" / "ohgi_regulations.md"
COLLECTION = "ohgi_regulations"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_VERSION = 1


@dataclass(frozen=True)
class IndexEmbedding:
    model: str
    function: Embeddings
    dimensions: int


class LocalPolicyEmbeddings(Embeddings):
    def __init__(self, runtime_dir: Path, *, local_files_only: bool):
        self._model = TextEmbedding(
            model_name=EMBEDDING_MODEL,
            cache_dir=str(runtime_dir / "embedding-models"),
            threads=2,
            local_files_only=local_files_only,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


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
    if (
        manifest["source_sha256"] != hashlib.sha256(REGULATIONS.read_bytes()).hexdigest()
        or manifest["embedding_model"] != model_name
        or (embedding is not None and manifest.get("embedding_dimensions") != embedding.dimensions)
        or manifest["version"] != INDEX_VERSION
    ):
        raise ValueError("규정 또는 임베딩 설정이 바뀌었습니다. 정책 인덱스를 다시 만드세요.")
    directory = runtime_dir / "policy-indexes" / manifest["directory"]
    if not (directory / "chroma.sqlite3").is_file():
        raise FileNotFoundError("정책 인덱스 파일을 찾을 수 없습니다.")
    index = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding.function if embedding else LocalPolicyEmbeddings(
            runtime_dir, local_files_only=True,
        ),
        persist_directory=str(directory),
        client_settings=ChromaSettings(anonymized_telemetry=False),
        create_collection_if_not_exists=False,
    )
    if len(index.get(include=[])["ids"]) != manifest["article_count"]:
        raise ValueError("정책 인덱스에 저장된 조문 수가 manifest와 다릅니다.")
    return index


def build_policy_index(runtime_dir: Path, *, embedding: IndexEmbedding | None = None) -> dict:
    source_hash = hashlib.sha256(REGULATIONS.read_bytes()).hexdigest()
    directory_name = uuid4().hex
    directory = runtime_dir / "policy-indexes" / directory_name
    directory.mkdir(parents=True)
    articles = load_articles()
    index = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding.function if embedding else LocalPolicyEmbeddings(
            runtime_dir, local_files_only=False,
        ),
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
        "embedding_model": embedding.model if embedding else EMBEDDING_MODEL,
        "article_count": len(ids),
    }
    if embedding is not None:
        manifest["embedding_dimensions"] = embedding.dimensions
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
