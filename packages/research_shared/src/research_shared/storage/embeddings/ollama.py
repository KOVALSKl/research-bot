import httpx


class OllamaDenseEmbedder:
    """Dense embeddings via Ollama /api/embed endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": texts},
            )
            response.raise_for_status()
            data = response.json()

        embeddings: list[list[float]] = data["embeddings"]
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Ollama returned {len(embeddings)} embeddings for {len(texts)} inputs"
            )
        return embeddings

    def probe_vector_size(self) -> int:
        return len(self.embed(["probe"])[0])
