import httpx


class OllamaLLMProvider:
    """Answer generation via Ollama chat endpoint (/api/chat)."""

    def __init__(
        self,
        model: str,
        base_url: str,
        system_prompt: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._system_prompt = system_prompt
        self._timeout = timeout_seconds

    def generate(self, question: str, context: str) -> str:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"},
        ]
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/api/chat",
                json={"model": self._model, "messages": messages, "stream": False},
            )
            response.raise_for_status()
            data = response.json()

        return data["message"]["content"]
