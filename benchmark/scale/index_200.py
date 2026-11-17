"""Create one OpenAI vector bundle and search it through Chroma or pgvector."""
import argparse
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

import anyio.to_thread
import chromadb
import httpx
import langsmith as ls
import numpy as np
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from openai import OpenAIError
from psycopg_pool import ConnectionPool
from pydantic import SecretStr

from app.core.config import PROJECT_ROOT
from app.services.policy_index import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, load_articles, openai_policy_embedding
from benchmark.langsmith_embeddings import EmbeddingUsage
from benchmark.langsmith_providers import required_setting


ARTIFACTS = PROJECT_ROOT / ".local" / "ohgi-200p"
BOOK = ARTIFACTS / "regulations-200p.md"
MATRIX = ARTIFACTS / "vectors.npy"
MANIFEST = ARTIFACTS / "vector-bundle.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


@contextmanager
def metered_embeddings(journal: Path | None = None):
    usage = EmbeddingUsage()
    lock = Lock()

    def record(response: httpx.Response) -> None:
        usage.record(response)
        if journal is not None and response.is_success and response.request.url.path.endswith("/embeddings"):
            tokens = response.json().get("usage", {}).get("total_tokens", 0)
            with lock, journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"tokens": tokens}) + "\n")

    with httpx.Client(timeout=60, event_hooks={"response": [record]}) as transport:
        key = SecretStr(required_setting("OPENAI_API_KEY"))
        model = openai_policy_embedding(key, http_client=transport)
        yield model.function, usage


@dataclass(frozen=True)
class Bundle:
    documents: list[Document]
    vectors: np.ndarray
    manifest: dict


def load_bundle() -> Bundle:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    documents = load_articles(BOOK)
    vectors = np.load(MATRIX, allow_pickle=False)
    if (
        digest(BOOK) != manifest["corpus_sha256"]
        or [doc.id for doc in documents] != manifest["article_ids"]
        or digest(MATRIX) != manifest["vector_sha256"]
        or manifest["embedding_model"] != EMBEDDING_MODEL
        or manifest["dimensions"] != EMBEDDING_DIMENSIONS
        or vectors.shape != (605, EMBEDDING_DIMENSIONS)
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("규정집, 조문 ID 또는 임베딩 파일이 manifest와 다릅니다.")
    return Bundle(documents, vectors, manifest)


def build_bundle() -> Bundle:
    if MANIFEST.is_file():
        return load_bundle()
    documents = load_articles(BOOK)
    ids = [doc.id for doc in documents]
    if ids != [f"제{i}조" for i in range(1, 606)]:
        raise ValueError("임베딩 전에 제1~605조 순서를 확인하세요.")
    start = perf_counter()
    vectors = []
    with metered_embeddings() as (embedding, usage):
        try:
            for offset in range(0, len(documents), 64):
                vectors.extend(embedding.embed_documents([
                    doc.page_content for doc in documents[offset:offset + 64]
                ]))
                print(f"임베딩 {len(vectors)}/{len(documents)} · {usage.snapshot()['tokens']} 입력 토큰", flush=True)
        except OpenAIError as exc:
            raise RuntimeError(f"OpenAI 임베딩 실패. {type(exc).__name__}") from None
    matrix = np.asarray(vectors, dtype="<f4")
    if matrix.shape != (605, EMBEDDING_DIMENSIONS) or not np.isfinite(matrix).all():
        raise ValueError("임베딩 응답의 행과 차원이 다릅니다.")
    pending = MATRIX.with_name("vectors.tmp.npy")
    np.save(pending, matrix, allow_pickle=False)
    pending.replace(MATRIX)
    manifest = {
        "corpus_sha256": digest(BOOK), "embedding_model": EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIMENSIONS, "article_ids": ids, "vector_sha256": digest(MATRIX),
        "build_embedding_usage": usage.snapshot(), "elapsed_seconds": round(perf_counter() - start, 3),
    }
    write_json(MANIFEST, manifest)
    return load_bundle()


def vector_literal(vector) -> str:
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


class ChromaStore:
    def __init__(self, bundle: Bundle):
        self.client = chromadb.PersistentClient(
            path=str(ARTIFACTS / "chroma"), settings=ChromaSettings(anonymized_telemetry=False),
        )
        name = "ohgi-" + bundle.manifest["vector_sha256"][:20]
        self.collection = self.client.get_or_create_collection(
            name=name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}},
        )
        stored = set(self.collection.get(include=[])["ids"])
        wanted = set(bundle.manifest["article_ids"])
        if not stored <= wanted:
            raise ValueError("Chroma에 다른 조문의 벡터가 포함돼 있습니다.")
        for start in range(0, len(bundle.documents), 64):
            offsets = [
                j for j in range(start, min(start + 64, len(bundle.documents)))
                if bundle.documents[j].id not in stored
            ]
            if offsets:
                docs = [bundle.documents[j] for j in offsets]
                self.collection.upsert(
                    ids=[doc.id for doc in docs], embeddings=bundle.vectors[offsets].tolist(),
                    documents=[doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs],
                )
        if set(self.collection.get(include=[])["ids"]) != wanted:
            raise ValueError("Chroma의 조문 ID가 임베딩 파일과 다릅니다.")
        actual = self.collection.get(ids=["제1조"], include=["embeddings"])["embeddings"]
        # Cosine indexes may round normalized float32 values by an ULP.
        np.testing.assert_allclose(actual[0], bundle.vectors[0], rtol=2e-6, atol=2e-8)

    def nearest(self, vector: list[float], k: int) -> list[str]:
        return self.collection.query(query_embeddings=[vector], n_results=k, include=[])["ids"][0]

    def lookup(self, ids: list[str]) -> list[str]:
        return self.collection.get(ids=ids, include=[])["ids"]


class PgVectorStore:
    def __init__(self, pool: ConnectionPool, bundle: Bundle):
        self.pool = pool
        with pool.connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS corpus_info (sha text PRIMARY KEY)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS policy_articles ("
                "ordinal integer PRIMARY KEY, article_id text NOT NULL UNIQUE, "
                "body text NOT NULL, embedding vector(3072) NOT NULL)"
            )
            old_hash = conn.execute("SELECT sha FROM corpus_info").fetchall()
            value = bundle.manifest["vector_sha256"]
            if old_hash and old_hash != [(value,)]:
                raise ValueError("실험 DB에 이미 다른 임베딩 데이터가 있습니다.")
            if not old_hash:
                with conn.cursor().copy(
                    "COPY policy_articles (ordinal, article_id, body, embedding) FROM STDIN"
                ) as copy:
                    for ordinal, (doc, vector) in enumerate(zip(bundle.documents, bundle.vectors, strict=True), 1):
                        copy.write_row((ordinal, doc.id, doc.page_content, vector_literal(vector)))
                conn.execute("INSERT INTO corpus_info (sha) VALUES (%s)", (value,))
            rows = conn.execute("SELECT article_id FROM policy_articles ORDER BY ordinal").fetchall()
            if [row[0] for row in rows] != bundle.manifest["article_ids"]:
                raise ValueError("pgvector의 조문 ID가 임베딩 파일과 다릅니다.")
            sample, = conn.execute("SELECT embedding::text FROM policy_articles WHERE ordinal = 1").fetchone()
            np.testing.assert_array_equal(np.array(json.loads(sample), dtype="<f4"), bundle.vectors[0])

    def nearest(self, vector: list[float], k: int) -> list[str]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT article_id FROM policy_articles ORDER BY embedding <=> %s::vector, ordinal LIMIT %s",
                (vector_literal(vector), k),
            )
            return [row[0] for row in rows]

    def lookup(self, ids: list[str]) -> list[str]:
        with self.pool.connection() as conn:
            return [row[0] for row in conn.execute(
                "SELECT article_id FROM policy_articles WHERE article_id = ANY(%s)", (ids,),
            )]


class MeasuredIndex:
    def __init__(self, store: ChromaStore | PgVectorStore, documents: list[Document], embedding):
        self.store = store
        self.documents = {doc.id: doc for doc in documents}
        self.embedding = embedding
        self._times = []
        self._lock = Lock()

    def times(self) -> list[float]:
        with self._lock:
            return self._times.copy()

    @ls.traceable(name="vector_lookup", process_inputs=lambda fields: {"k": fields["k"]})
    def _nearest(self, vector: list[float], k: int) -> list[str]:
        start = perf_counter()
        try:
            return self.store.nearest(vector, k)
        finally:
            with self._lock:
                self._times.append(round((perf_counter() - start) * 1000, 3))

    def query(self, text: str, k: int) -> list[Document]:
        vector = self.embedding.embed_query(text)
        return [self.documents[article] for article in self._nearest(vector, k)]

    async def asimilarity_search(self, query: str, k: int) -> list[Document]:
        return await anyio.to_thread.run_sync(self.query, query, k)

    async def aget_by_ids(self, ids: list[str]) -> list[Document]:
        found = await anyio.to_thread.run_sync(self.store.lookup, list(ids))
        return [self.documents[article] for article in found]


@contextmanager
def open_store(backend: str, bundle: Bundle) -> Iterator[ChromaStore | PgVectorStore]:
    if backend == "chroma":
        yield ChromaStore(bundle)
    elif backend == "pgvector":
        with ConnectionPool(
            kwargs={"host": required_setting("OHGI_PGHOST"), "port": 46581, "dbname": "postgres"},
            min_size=1, max_size=4, timeout=8,
        ) as pool:
            yield PgVectorStore(pool, bundle)
    else:
        raise ValueError(f"Unknown backend {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["embed", "check-chroma", "check-pgvector"])
    args = parser.parse_args()
    if args.stage == "embed":
        bundle = build_bundle()
        print(json.dumps({k: v for k, v in bundle.manifest.items() if k != "article_ids"}), flush=True)
    else:
        bundle = load_bundle()
        with open_store(args.stage.removeprefix("check-"), bundle) as store:
            first = store.nearest(bundle.vectors[0].tolist(), 3)
            assert first[0] == "제1조"
            print(json.dumps({"stage": args.stage, "nearest_to_article_1": first, "documents": len(bundle.documents)},
                             ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
