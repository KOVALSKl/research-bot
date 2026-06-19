import re
from pathlib import Path

import pymupdf

from research_shared.ingestion.protocols import ParsedDocument, ParsedPage

_CHAPTER_PATTERNS = (
    re.compile(r"^\d+(\.\d+)*\s+"),
    re.compile(r"^Глава\s", re.IGNORECASE),
    re.compile(r"^Chapter\s", re.IGNORECASE),
)
_AUTHOR_SPLIT = re.compile(r";|,|\s+and\s+|\s+и\s+", re.IGNORECASE)


class PyMuPDFParser:
    """PDF parser based on PyMuPDF (pymupdf): per-page text + basic metadata."""

    def parse(self, path: str | Path) -> ParsedDocument:
        path = Path(path)

        pages: list[ParsedPage] = []
        pdf_metadata: dict = {}
        current_chapter: str | None = None

        with pymupdf.open(path) as doc:
            pdf_metadata = dict(doc.metadata or {})
            for index, page in enumerate(doc, start=1):
                try:
                    text = page.get_text("text") or ""
                except Exception:
                    # Defensive: a single corrupt page must not abort the document.
                    text = ""

                try:
                    detected = self._detect_chapter_heading(page)
                except Exception:
                    detected = None
                if detected:
                    current_chapter = detected

                pages.append(
                    ParsedPage(
                        page=index,
                        text=text.strip(),
                        chapter=current_chapter,
                    )
                )

        title = self._resolve_title(pdf_metadata, path)
        authors = self._parse_authors(pdf_metadata)

        return ParsedDocument(
            title=title,
            pages=pages,
            metadata={
                "source_path": str(path),
                "filename": path.name,
                "page_count": len(pages),
                "pdf_metadata": pdf_metadata,
                "authors": authors,
            },
        )

    @staticmethod
    def _resolve_title(pdf_metadata: dict, path: Path) -> str:
        title = (pdf_metadata.get("title") or "").strip()
        if title:
            return title
        return path.stem

    @staticmethod
    def _parse_authors(pdf_metadata: dict) -> list[str]:
        author = (pdf_metadata.get("author") or "").strip()
        if not author:
            return []

        authors: list[str] = []
        for part in _AUTHOR_SPLIT.split(author):
            name = " ".join(part.split())
            if name:
                authors.append(name)
        return authors

    @staticmethod
    def _detect_chapter_heading(page: pymupdf.Page) -> str | None:
        try:
            blocks = page.get_text("dict")
        except Exception:
            return None

        sizes: list[float] = []
        lines: list[tuple[str, float]] = []

        for block in blocks.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join(span.get("text", "") for span in spans).strip()
                if not line_text:
                    continue
                max_size = max((span.get("size", 0) for span in spans), default=0)
                if max_size > 0:
                    sizes.append(max_size)
                lines.append((line_text, max_size))

        if not sizes:
            return None

        median = sorted(sizes)[len(sizes) // 2]
        threshold = median * 1.2

        for text, size in lines:
            if size < threshold:
                continue
            if len(text) >= 120:
                continue
            if text.endswith("."):
                continue
            if any(pattern.search(text) for pattern in _CHAPTER_PATTERNS):
                return text
            return text

        return None
