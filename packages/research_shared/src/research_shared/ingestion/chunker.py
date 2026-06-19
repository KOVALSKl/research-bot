from research_shared.config.settings import Settings
from research_shared.domain.models import ResearchChunk
from research_shared.ingestion.protocols import ParsedDocument

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker:
    """Splits document text into overlapping chunks bound to their source page.

    Each page is chunked independently so that the ``page`` metadata stays
    accurate for citation. ``chunk_index`` is assigned sequentially across the
    whole document.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or Settings()
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap

    def chunk(self, document: ParsedDocument, research_id: str) -> list[ResearchChunk]:
        chunks: list[ResearchChunk] = []
        chunk_index = 0

        for page in document.pages:
            text = (page.text or "").strip()
            if not text:
                continue

            for piece in self._split_text(text):
                piece = piece.strip()
                if not piece:
                    continue
                chunks.append(
                    ResearchChunk(
                        research_id=research_id,
                        title=document.title,
                        text=piece,
                        source_path=document.metadata.get("source_path"),
                        authors=document.metadata.get("authors", []),
                        chapter=page.chapter,
                        metadata={
                            "page": page.page,
                            "chunk_index": chunk_index,
                            "chapter": page.chapter,
                        },
                    )
                )
                chunk_index += 1

        return chunks

    def _split_text(self, text: str) -> list[str]:
        atoms = self._recursive_split(text, _SEPARATORS)
        return self._merge(atoms)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self._chunk_size:
            return [text] if text else []

        separator = separators[0]
        remaining = separators[1:]

        if separator == "":
            return [
                text[i : i + self._chunk_size]
                for i in range(0, len(text), self._chunk_size)
            ]

        pieces = text.split(separator)
        result: list[str] = []
        for piece in pieces:
            if not piece:
                continue
            if len(piece) <= self._chunk_size:
                result.append(piece)
            elif remaining:
                result.extend(self._recursive_split(piece, remaining))
            else:
                result.extend(
                    piece[i : i + self._chunk_size]
                    for i in range(0, len(piece), self._chunk_size)
                )
        return result

    def _merge(self, atoms: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        total = 0
        join = " "
        join_len = len(join)

        for atom in atoms:
            atom_len = len(atom)
            addition = atom_len + (join_len if current else 0)
            if current and total + addition > self._chunk_size:
                chunks.append(join.join(current))
                while current and total > self._chunk_overlap:
                    removed = current.pop(0)
                    total -= len(removed) + (join_len if current else 0)
            total += atom_len + (join_len if current else 0)
            current.append(atom)

        if current:
            chunks.append(join.join(current))

        return chunks
