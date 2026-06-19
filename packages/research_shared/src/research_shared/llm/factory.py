import importlib

from research_shared.config.settings import Settings
from research_shared.llm.huggingface import HuggingFaceInferenceProvider
from research_shared.llm.ollama import OllamaLLMProvider
from research_shared.llm.prompts import get_rag_system_prompt
from research_shared.llm.protocols import LLMProvider


def _load_custom_provider(module_path: str, settings: Settings) -> LLMProvider:
    """Load a custom LLM provider class from a dotted path (e.g. ``my_app.llm.CustomLLM``)."""
    path = module_path.strip()
    if not path:
        raise ValueError(
            "llm_provider_module is required when llm_provider=custom and llm_enabled=true"
        )

    parts = path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid llm_provider_module {path!r}: expected dotted path like 'my_app.llm.CustomLLM'"
        )

    module_name, class_name = parts
    try:
        module = importlib.import_module(module_name)
        provider_cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Failed to load custom LLM provider {path!r}: {exc}") from exc

    instance = provider_cls(settings)
    if not isinstance(instance, LLMProvider):
        raise TypeError(f"{path} does not implement LLMProvider")

    return instance


def create_llm_provider(settings: Settings | None = None) -> LLMProvider | None:
    """Build an LLM provider, or ``None`` when LLM generation is disabled.

    Built-in providers: ``huggingface`` (default), ``ollama``, or ``custom`` via
    ``llm_provider_module``.
    """
    settings = settings or Settings()

    if not settings.llm_enabled:
        return None

    provider = settings.llm_provider

    system_prompt = get_rag_system_prompt(settings)

    if provider == "huggingface":
        if not settings.hf_api_token:
            raise ValueError(
                "HF_API_TOKEN is required when llm_enabled=true and llm_provider=huggingface"
            )
        return HuggingFaceInferenceProvider(
            model=settings.hf_model,
            api_token=settings.hf_api_token,
            system_prompt=system_prompt,
            timeout_seconds=settings.hf_timeout_seconds,
        )

    if provider == "ollama":
        return OllamaLLMProvider(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_url,
            system_prompt=system_prompt,
            timeout_seconds=settings.ollama_timeout_seconds,
        )

    if provider == "custom":
        return _load_custom_provider(settings.llm_provider_module, settings)

    raise ValueError(f"Unknown llm_provider: {provider!r}")
