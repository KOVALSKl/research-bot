from __future__ import annotations

from urllib.parse import urljoin

import httpx

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


async def fetch_with_redirects(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_redirects: int = 10,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """GET *url*, following redirects manually when httpx stops early."""
    request_headers = headers or {}
    redirect_count = 0
    current_url = url

    while True:
        response = await client.get(
            current_url,
            follow_redirects=False,
            headers=request_headers,
        )
        if response.status_code not in _REDIRECT_STATUSES:
            return response

        if redirect_count >= max_redirects:
            raise httpx.TooManyRedirects(
                f"Exceeded max redirects ({max_redirects}) for {url}",
                request=response.request,
            )

        location = response.headers.get("Location")
        if not location:
            return response

        redirect_count += 1
        current_url = urljoin(str(response.url), location)
