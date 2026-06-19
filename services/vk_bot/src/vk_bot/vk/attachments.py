from __future__ import annotations

from collections.abc import Awaitable, Callable

from vk_bot.domain import Attachment

DocUrlResolver = Callable[[int, int], Awaitable[str | None]]


def _normalize_raw(raw: list | dict | None) -> list[dict]:
  if raw is None:
    return []
  if isinstance(raw, dict):
    return [raw]
  if isinstance(raw, list):
    return [item for item in raw if isinstance(item, dict)]
  return []


def _doc_filename(doc: dict) -> str:
  ext = str(doc.get("ext", "")).lower()
  title = str(doc.get("title", "document"))
  if ext and not title.lower().endswith(f".{ext}"):
    return f"{title}.{ext}"
  return title


async def parse_doc_attachments(
  raw: list | dict | None,
  *,
  doc_url_resolver: DocUrlResolver | None = None,
  resolve_urls: bool = True,
) -> list[Attachment]:
  """Parse VK doc attachments and keep PDF files only."""
  attachments: list[Attachment] = []
  for item in _normalize_raw(raw):
    if item.get("type") != "doc":
      continue
    doc = item.get("doc")
    if not isinstance(doc, dict):
      continue

    ext = str(doc.get("ext", "")).lower()
    if ext != "pdf":
      continue

    owner_id = int(doc.get("owner_id") or 0)
    doc_id = int(doc.get("id") or 0)
    url = str(doc.get("url") or doc.get("access_url") or "")

    if not url and resolve_urls and doc_url_resolver is not None and owner_id and doc_id:
      resolved = await doc_url_resolver(owner_id, doc_id)
      if resolved:
        url = resolved

    if not url and not (owner_id and doc_id):
      continue

    attachments.append(
      Attachment(
        filename=_doc_filename(doc),
        url=url,
        ext=ext,
        size=int(doc.get("size", 0) or 0),
        owner_id=owner_id,
        doc_id=doc_id,
      )
    )
  return attachments
