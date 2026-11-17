"""An evaluation index must be opened with the model and size it was built for."""

import pytest
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

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
