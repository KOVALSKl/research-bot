import uvicorn

from research_shared.config.settings import get_settings

from core_api.app import create_app


def run() -> None:
    settings = get_settings()
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.core_api_host,
        port=settings.core_api_port,
    )


if __name__ == "__main__":
    run()
