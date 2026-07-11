from src.embedding.simple import HashEmbeddingProvider


def test_hash_embedding_provider_is_deterministic():
    provider = HashEmbeddingProvider(dimensions=8)

    first = provider.embed_query("脑出血 CT")
    second = provider.embed_query("脑出血 CT")

    assert first == second
    assert len(first) == 8
