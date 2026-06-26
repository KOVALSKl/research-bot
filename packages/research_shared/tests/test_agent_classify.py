import pytest

from research_shared.agents.classify import classify_by_rules, classify_request
from research_shared.agents.models import AgentAskRequest
from research_shared.config.settings import Settings


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Оцени идею: использовать трансформеры для прогноза цен", "idea_evaluation"),
        ("Насколько перспективно исследовать мембранные материалы?", "idea_evaluation"),
        ("Предлагаю исследовать влияние LLM на образование", "idea_evaluation"),
        ("Evaluate my idea about graph neural networks for fraud", "idea_evaluation"),
        ("How promising is research on quantum error correction?", "idea_evaluation"),
        ("Что такое финансовая пирамида?", None),
        ("Explain transformer architecture", None),
    ],
)
def test_classify_by_rules(message: str, expected: str | None) -> None:
    assert classify_by_rules(message) == expected


def test_classify_request_explicit_modes() -> None:
    assert (
        classify_request(AgentAskRequest(message="text", mode="question"))
        == "question"
    )
    assert (
        classify_request(AgentAskRequest(message="text", mode="idea_evaluation"))
        == "idea_evaluation"
    )


def test_classify_request_auto_detects_idea() -> None:
    request = AgentAskRequest(
        message="Насколько перспективна идея применить GNN к транзакциям?",
        mode="auto",
    )
    assert classify_request(request) == "idea_evaluation"


def test_classify_request_auto_defaults_to_question() -> None:
    request = AgentAskRequest(
        message="Какие методы используются в исследованиях пирамид?",
        mode="auto",
    )
    assert classify_request(request) == "question"
