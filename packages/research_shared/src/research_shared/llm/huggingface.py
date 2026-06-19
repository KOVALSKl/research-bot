import httpx

_HF_CHAT_COMPLETIONS_URL = "https://router.huggingface.co/v1/chat/completions"


class HuggingFaceInferenceProvider:
    """Answer generation via Hugging Face Inference API (chat completions)."""

    def __init__(
        self,
        model: str,
        api_token: str,
        system_prompt: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._model = model
        self._api_token = api_token
        self._system_prompt = system_prompt
        self._timeout = timeout_seconds

    def generate(self, question: str, context: str) -> str:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"},
        ]
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                _HF_CHAT_COMPLETIONS_URL,
                headers={
                    "Authorization": f"Bearer {self._api_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()

        choices = data.get("choices")
        if not choices:
            raise RuntimeError(f"HF Inference API returned no choices: {data}")

        content = choices[0].get("message", {}).get("content")
        if content is None:
            raise RuntimeError(f"HF Inference API returned empty content: {data}")

        return content
