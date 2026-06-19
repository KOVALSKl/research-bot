from dataclasses import dataclass, field


@dataclass(frozen=True)
class Attachment:
  filename: str
  url: str
  ext: str = ""
  size: int = 0
  owner_id: int = 0
  doc_id: int = 0


@dataclass(frozen=True)
class IncomingMessage:
  """Normalized inbound VK message."""

  user_id: int
  peer_id: int
  text: str = ""
  attachments: list[Attachment] = field(default_factory=list)
  from_id: int = 0
  is_outgoing: bool = False
  message_id: int = 0
  conversation_message_id: int = 0


@dataclass(frozen=True)
class OutgoingMessage:
  peer_id: int
  text: str
