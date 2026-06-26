from research_shared.config.settings import Settings
from research_shared.llm.prompts import (
    AGENT_IDEA_EVAL_PROMPT,
    AGENT_IDEA_QUERY_REFORMULATION_PROMPT,
    AGENT_QUESTION_PROMPT,
    AGENT_REACT_SYSTEM_PROMPT,
    DEFAULT_RAG_SYSTEM_PROMPT,
    get_agent_idea_eval_prompt,
    get_agent_idea_query_reformulation_prompt,
    get_agent_question_prompt,
    get_agent_react_system_prompt,
)


def test_get_agent_question_prompt_default() -> None:
    settings = Settings(_env_file=None, agent_question_prompt="")
    assert get_agent_question_prompt(settings) == AGENT_QUESTION_PROMPT


def test_get_agent_question_prompt_override() -> None:
    custom = "Custom agent prompt."
    settings = Settings(_env_file=None, agent_question_prompt=custom)
    assert get_agent_question_prompt(settings) == custom


def test_agent_prompt_mentions_external_citations() -> None:
    prompt = AGENT_QUESTION_PROMPT
    assert "[E1]" in prompt or "[E n]" in prompt
    assert "[1]" in prompt


def test_agent_prompt_differs_from_rag_prompt() -> None:
    assert AGENT_QUESTION_PROMPT != DEFAULT_RAG_SYSTEM_PROMPT


def test_agent_prompt_v14_structure() -> None:
    prompt = AGENT_QUESTION_PROMPT
    assert "Прямой ответ" in prompt
    assert "Обзор релевантных исследований" in prompt
    assert "Синтез" in prompt
    assert "Пробелы" in prompt


def test_agent_prompt_v15_evidence_requirements() -> None:
    prompt = AGENT_QUESTION_PROMPT
    assert "минимум 4 предложения" in prompt
    assert "см. источники" in prompt.lower() or "см. источник" in prompt.lower()
    assert "конкрет" in prompt.lower()


def test_agent_prompt_forbids_shallow_answers() -> None:
    prompt = AGENT_QUESTION_PROMPT.lower()
    assert "да, исследования есть" in prompt


def test_agent_prompt_ignores_irrelevant_sources() -> None:
    prompt = AGENT_QUESTION_PROMPT.lower()
    assert "игнорируй" in prompt


def test_idea_eval_prompt_requires_json_and_citations() -> None:
    prompt = AGENT_IDEA_EVAL_PROMPT
    assert "evidence_for" in prompt
    assert "[E1]" in prompt or "[En]" in prompt
    assert "confidence" in prompt
    assert "аргумент за" not in prompt.lower() or "запрещено" in prompt.lower()


def test_idea_eval_prompt_has_no_copyable_placeholders() -> None:
    prompt = AGENT_IDEA_EVAL_PROMPT
    assert "аргумент за с [n] или [En] и конкретным фактом" not in prompt


def test_react_system_prompt_lists_whitelist_tools() -> None:
    prompt = AGENT_REACT_SYSTEM_PROMPT
    for tool in (
        "local_hybrid_search",
        "external_literature_search",
        "reformulate_queries",
        "finish",
    ):
        assert tool in prompt
    assert "thought" in prompt
    assert "action_input" in prompt


def test_get_agent_react_system_prompt_default() -> None:
    settings = Settings(_env_file=None, agent_react_system_prompt="")
    assert get_agent_react_system_prompt(settings) == AGENT_REACT_SYSTEM_PROMPT


def test_idea_query_reformulation_prompt_mentions_english() -> None:
    prompt = AGENT_IDEA_QUERY_REFORMULATION_PROMPT
    assert "англ" in prompt.lower() or "english" in prompt.lower()


def test_get_agent_idea_eval_prompt_override() -> None:
    custom = "Custom idea prompt."
    settings = Settings(_env_file=None, agent_idea_eval_prompt=custom)
    assert get_agent_idea_eval_prompt(settings) == custom


def test_get_agent_idea_query_reformulation_prompt_default() -> None:
    settings = Settings(_env_file=None, agent_idea_query_reformulation_prompt="")
    assert get_agent_idea_query_reformulation_prompt(settings) == AGENT_IDEA_QUERY_REFORMULATION_PROMPT
