from __future__ import annotations

import hashlib
import re

from research_shared.literature.models import ExternalPaper

_FILENAME_SAFE = re.compile(r"[^\w\s\-().]", re.UNICODE)


def build_external_cache_key(paper: ExternalPaper) -> str:
    if paper.doi:
        raw = f"doi:{paper.doi.strip().lower()}"
    elif paper.pdf_url:
        raw = f"pdf:{paper.pdf_url.strip()}"
    else:
        raw = f"{paper.source}:{paper.title.strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def external_pdf_filename(paper: ExternalPaper) -> str:
    base = _FILENAME_SAFE.sub("", paper.title).strip() or "external_paper"
    base = re.sub(r"\s+", "_", base)
    if len(base) > 80:
        base = base[:80]
    return f"{base}.pdf"
