from fastembed import SparseTextEmbedding, TextEmbedding

from research_shared.storage.protocols import DenseEmbedder, SparseEncoder


class FastEmbedDenseEmbedder:
    def __init__(self, model_name: str) -> None:
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def probe_vector_size(self) -> int:
        return len(self.embed(["probe"])[0])


class FastEmbedSparseEncoder:
    def __init__(self, model_name: str) -> None:
        self._model = SparseTextEmbedding(model_name=model_name)

    def encode(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        results: list[tuple[list[int], list[float]]] = []
        for embedding in self._model.embed(texts):
            indices = embedding.indices.tolist()
            values = embedding.values.tolist()
            results.append((indices, values))
        return results
