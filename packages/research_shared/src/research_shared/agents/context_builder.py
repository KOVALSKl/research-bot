def build_agent_context(local_context: str, external_context: str) -> str:
    sections: list[str] = []
    local = local_context.strip()
    external = external_context.strip()

    if local:
        sections.append(f"Локальные источники:\n{local}")
    if external:
        sections.append(f"Внешние публикации:\n{external}")

    return "\n\n".join(sections)
