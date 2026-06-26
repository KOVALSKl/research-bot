"""Yandex AI Studio LLM provider (openai SDK + Responses API)."""

from __future__ import annotations

import openai


class YandexAIStudioProvider:
    """Answer generation via Yandex AI Studio Responses API."""

    def __init__(
        self,
        model: str,
        api_key: str,
        folder_id: str,
        base_url: str,
        system_prompt: str,
        temperature: float = 0.3,
        max_output_tokens: int = 2000,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._model = model
        self._folder_id = folder_id
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            project=folder_id,
            timeout=timeout_seconds,
        )

    def generate(self, question: str, context: str) -> str:
        response = self._client.responses.create(
            model=f"gpt://{self._folder_id}/{self._model}",
            temperature=self._temperature,
            instructions=self._system_prompt,
            input=(
                f"<documents>\n{context}\n</documents>\n\n"
                f"<user_query>\n{question}\n</user_query>"
            ),
            max_output_tokens=self._max_output_tokens,
        )
        return response.output_text
