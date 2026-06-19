from vk_bot.config import VkBotSettings
from vk_bot.transport.protocols import VkTransport
from vk_bot.vk.message_enricher import MessageEnricher


def create_vk_transport(
  settings: VkBotSettings,
  message_enricher: MessageEnricher | None = None,
) -> VkTransport:
  transport = settings.vk_transport

  if transport == "long_polling":
    from vk_bot.transport.long_polling import VkLongPollingTransport

    return VkLongPollingTransport(settings, message_enricher=message_enricher)

  if transport == "callback":
    from vk_bot.transport.callback_api import VkCallbackTransport

    return VkCallbackTransport(settings, message_enricher=message_enricher)

  raise ValueError(f"Unknown vk_transport: {transport!r}")
