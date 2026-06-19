from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    def generate(self, question: str, context: str) -> str: ...
