"""Build and meter the course's OpenAI embedding index for the notebook."""

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import httpx
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAIError
from pydantic import SecretStr

from app.core.config import PROJECT_ROOT
from app.services.policy_index import IndexEmbedding, REGULATIONS, build_policy_index, open_policy_index
from benchmark.langsmith_providers import required_setting


RESULTS_DIR = PROJECT_ROOT / ".local" / "langsmith-eval"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072


class EmbeddingUsage:
    def __init__(self):
        self._lock = Lock()
        self._requests = 0
        self._tokens = 0

    def record(self, response: httpx.Response) -> None:
        if not response.is_success or not response.request.url.path.endswith("/embeddings"):
            return
        response.read()
        tokens = response.json().get("usage", {}).get("total_tokens", 0)
        with self._lock:
            self._requests += 1
            self._tokens += tokens

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"requests": self._requests, "tokens": self._tokens}


@dataclass(frozen=True)
class EvaluationIndex:
    runtime_dir: Path
    embedding: IndexEmbedding
    usage: EmbeddingUsage
    build_usage: dict[str, int]

    def load(self, runtime_dir: Path) -> Chroma | None:
        return open_policy_index(runtime_dir, embedding=self.embedding)


@contextmanager
def openai_evaluation_index() -> Iterator[EvaluationIndex]:
    source_hash = hashlib.sha256(REGULATIONS.read_bytes()).hexdigest()[:12]
    runtime_dir = RESULTS_DIR / "index-openai-3-large" / source_hash
    usage = EmbeddingUsage()
    # Chroma's async search runs its synchronous embed_query in an executor.
    with httpx.Client(timeout=60, event_hooks={"response": [usage.record]}) as transport:
        function = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
            api_key=SecretStr(required_setting("OPENAI_API_KEY")),
            base_url="https://api.openai.com/v1",
            http_client=transport,
            request_timeout=60,
            max_retries=1,
        )
        embedding = IndexEmbedding(EMBEDDING_MODEL, function, EMBEDDING_DIMENSIONS)
        try:
            if open_policy_index(runtime_dir, embedding=embedding) is None:
                build_policy_index(runtime_dir, embedding=embedding)
        except OpenAIError as error:
            raise RuntimeError(f"OpenAI 임베딩 인덱스 생성에 실패했습니다. {type(error).__name__}") from None
        yield EvaluationIndex(runtime_dir, embedding, usage, usage.snapshot())
