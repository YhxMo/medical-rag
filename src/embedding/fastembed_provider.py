"""Local ONNX embeddings for the replacement English corpus."""
from src.embedding.base import EmbeddingProvider


class FastEmbedProvider(EmbeddingProvider):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", cache_dir="models/fastembed", threads=4):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self._model = None

    def _load_model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir,
                                        threads=self.threads)
        return self._model

    def embed_texts(self, texts):
        if not texts:
            return []
        return [vector.tolist() for vector in self._load_model().passage_embed(texts, batch_size=32)]

    def embed_query(self, query):
        return next(iter(self._load_model().query_embed(query))).tolist()
