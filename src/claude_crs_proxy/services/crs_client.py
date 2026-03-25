from __future__ import annotations

from typing import Any, Dict

import httpx

from src.claude_crs_proxy.config import settings


class CRSClient:
    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or settings.crs_base_url).rstrip("/")
        self.timeout = timeout or settings.request_timeout_seconds

    def build_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    async def post_json(self, path: str, *, headers: Dict[str, str], json_body: Dict[str, Any]) -> httpx.Response:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await client.post(self.build_url(path), headers=headers, json=json_body)

    async def post_stream(self, path: str, *, headers: Dict[str, str], json_body: Dict[str, Any]) -> httpx.Response:
        client = httpx.AsyncClient(timeout=self.timeout)
        request = client.build_request("POST", self.build_url(path), headers=headers, json=json_body)
        response = await client.send(request, stream=True)
        response._client = client
        return response
