from __future__ import annotations

import re
from dataclasses import dataclass

from research_shared.domain.models import Citation, ExternalSourceFileRef, SourceFileRef
from research_shared.literature.cache_keys import build_external_cache_key, external_pdf_filename
from research_shared.literature.models import ExternalPaper
from research_shared.logging_config import get_logger
from research_shared.rag.citations import build_source_files

logger = get_logger(__name__)

_CITATION_MARKER = re.compile(r"\[(E)?(\d+)\]")


@dataclass(frozen=True)
class CitedSources:
    local: list[Citation]
    external: list[ExternalPaper]
    local_indices: list[int]
    external_indices: list[int]
    source_files: list[SourceFileRef]
    external_source_files: list[ExternalSourceFileRef]


def build_external_source_files(
    external: list[ExternalPaper],
    external_indices: list[int],
) -> list[ExternalSourceFileRef]:
    refs: list[ExternalSourceFileRef] = []
    for index, paper in zip(external_indices, external, strict=True):
        if not paper.pdf_url:
            continue
        refs.append(
            ExternalSourceFileRef(
                external_index=index,
                title=paper.title,
                cache_key=build_external_cache_key(paper),
                filename=external_pdf_filename(paper),
                pdf_url=paper.pdf_url,
                display_name=paper.title,
            )
        )
    return refs


def extract_cited_sources(
    answer: str,
    local: list[Citation],
    external: list[ExternalPaper],
) -> CitedSources:
    """Map inline ``[n]`` / ``[En]`` markers in ``answer`` to deliverable sources."""
    if not answer.strip():
        return _fallback(local, external)

    local_order: list[int] = []
    external_order: list[int] = []
    seen_local: set[int] = set()
    seen_external: set[int] = set()

    for match in _CITATION_MARKER.finditer(answer):
        is_external = match.group(1) is not None
        index = int(match.group(2))
        if is_external:
            if index < 1 or index > len(external):
                logger.warning(
                    "Out-of-range external citation marker",
                    extra={
                        "event": "cited_sources.out_of_range",
                        "index": index,
                        "external_count": len(external),
                    },
                )
                continue
            if index not in seen_external:
                seen_external.add(index)
                external_order.append(index)
            continue

        if index < 1 or index > len(local):
            logger.warning(
                "Out-of-range local citation marker",
                extra={
                    "event": "cited_sources.out_of_range",
                    "index": index,
                    "local_count": len(local),
                },
            )
            continue
        if index not in seen_local:
            seen_local.add(index)
            local_order.append(index)

    if not local_order and not external_order:
        return _fallback(local, external)

    cited_local = [local[index - 1] for index in local_order]
    cited_external = [external[index - 1] for index in external_order]
    return CitedSources(
        local=cited_local,
        external=cited_external,
        local_indices=local_order,
        external_indices=external_order,
        source_files=build_source_files(cited_local),
        external_source_files=build_external_source_files(cited_external, external_order),
    )


def _fallback(local: list[Citation], external: list[ExternalPaper]) -> CitedSources:
    external_indices = list(range(1, len(external) + 1))
    return CitedSources(
        local=list(local),
        external=list(external),
        local_indices=list(range(1, len(local) + 1)),
        external_indices=external_indices,
        source_files=build_source_files(local),
        external_source_files=build_external_source_files(external, external_indices),
    )
