from vk_bot.config import VkBotSettings


def test_vk_ask_default_limit_defaults_to_ten() -> None:
    settings = VkBotSettings(_env_file=None)
    assert settings.vk_ask_default_limit == 10
