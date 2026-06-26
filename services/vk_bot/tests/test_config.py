from vk_bot.config import VkBotSettings


def test_vk_ask_default_limit_defaults_to_ten() -> None:
    settings = VkBotSettings(_env_file=None)
    assert settings.vk_ask_default_limit == 10


def test_json_array_list_env_var_parsed(monkeypatch) -> None:
    monkeypatch.setenv("VK_LIST_COMMAND_PREFIXES", '["/list","/research"]')
    settings = VkBotSettings(_env_file=None)
    assert settings.vk_list_command_prefixes == ["/list", "/research"]


def test_csv_list_env_vars_are_parsed_without_json(monkeypatch) -> None:
    monkeypatch.setenv("VK_LIST_COMMAND_PREFIXES", "/list,/исследования,/research")
    monkeypatch.setenv("VK_ASK_COMMAND_PREFIXES", "/ask,/вопрос,?,вопрос:")
    monkeypatch.setenv("VK_GREETING_KEYWORDS", "привет,старт,/start")

    settings = VkBotSettings(_env_file=None)

    assert settings.vk_list_command_prefixes == ["/list", "/исследования", "/research"]
    assert settings.vk_ask_command_prefixes == ["/ask", "/вопрос", "?", "вопрос:"]
    assert settings.vk_greeting_keywords == ["привет", "старт", "/start"]


def test_empty_csv_list_env_var_yields_empty_list(monkeypatch) -> None:
    monkeypatch.setenv("VK_LIST_COMMAND_PREFIXES", "")
    settings = VkBotSettings(_env_file=None)
    assert settings.vk_list_command_prefixes == []


def test_retry_settings_accept_arch_env_names(monkeypatch) -> None:
    monkeypatch.setenv("VK_CORE_API_RETRY_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("VK_CORE_API_RETRY_BASE_DELAY_SECONDS", "2.5")
    settings = VkBotSettings(_env_file=None)
    assert settings.vk_core_api_retry_max == 5
    assert settings.vk_core_api_retry_backoff_seconds == 2.5


def test_retry_settings_accept_legacy_env_names(monkeypatch) -> None:
    monkeypatch.setenv("VK_CORE_API_RETRY_MAX", "4")
    monkeypatch.setenv("VK_CORE_API_RETRY_BACKOFF_SECONDS", "0.5")
    settings = VkBotSettings(_env_file=None)
    assert settings.vk_core_api_retry_max == 4
    assert settings.vk_core_api_retry_backoff_seconds == 0.5
