"""An evaluation index must be opened with the model and size it was built for."""

import pytest
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from pydantic import SecretStr

from app.core.config import Settings
from app.services import policy_index
from app.services.policy_index import IndexEmbedding, build_policy_index, open_policy_index


def test_explicit_index_rejects_a_different_embedding_model_or_size(tmp_path):
    function = DeterministicFakeEmbedding(size=4)
    embedding = IndexEmbedding("test-embedding-model", function, 4)
    manifest = build_policy_index(tmp_path, embedding=embedding)

    index = open_policy_index(tmp_path, embedding=embedding)
    vectors = index.get(ids=["제1조"], include=["embeddings"])["embeddings"]
    assert manifest["embedding_model"] == embedding.model
    assert manifest["embedding_dimensions"] == 4
    assert len(vectors[0]) == 4

    with pytest.raises(ValueError, match="임베딩 설정"):
        open_policy_index(tmp_path)
    with pytest.raises(ValueError, match="임베딩 설정"):
        open_policy_index(tmp_path, embedding=IndexEmbedding(embedding.model, function, 3))


def test_default_index_build_and_open_use_openai(tmp_path, monkeypatch):
    calls = []

    def embedding_factory(*, model, dimensions, base_url, **kwargs):
        calls.append((model, dimensions, base_url))
        return DeterministicFakeEmbedding(size=dimensions)

    settings = Settings(_env_file=None, openai_api_key=SecretStr("test-key"))
    monkeypatch.setattr(policy_index, "get_settings", lambda: settings)
    monkeypatch.setattr(policy_index, "OpenAIEmbeddings", embedding_factory)

    manifest = build_policy_index(tmp_path)
    index = open_policy_index(tmp_path)

    expected = ("text-embedding-3-large", 3072, "https://api.openai.com/v1")
    assert calls == [expected, expected]
    assert manifest["embedding_model"] == expected[0]
    assert manifest["embedding_dimensions"] == expected[1]
    assert index.similarity_search("영업시간", k=1)
    assert not (tmp_path / "embedding-models").exists()


def test_missing_openai_key_fails_before_creating_index(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, openai_api_key=None)
    monkeypatch.setattr(policy_index, "get_settings", lambda: settings)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_policy_index(tmp_path)
    assert not list(tmp_path.iterdir())
