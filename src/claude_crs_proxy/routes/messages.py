from __future__ import annotations

import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

from src.claude_crs_proxy.config import settings
from src.claude_crs_proxy.schemas import MessagesRequest, TokenCountRequest
from src.claude_crs_proxy.services.auth import require_bearer_api_key
from src.claude_crs_proxy.services.crs_client import CRSClient
from src.claude_crs_proxy.services.field_mapper import merge_request_body
from src.claude_crs_proxy.services.model_routing import maybe_remap_model
from src.claude_crs_proxy.services.streaming import streaming_response

router = APIRouter()
client = CRSClient()


def build_forward_headers(request: Request) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for header_name in ["authorization", "anthropic-version", "anthropic-beta", "x-request-id", "x-stainless-arch", "x-stainless-lang", "x-stainless-os", "x-stainless-package-version", "x-stainless-runtime", "x-stainless-runtime-version", "x-stainless-timeout", "x-app", "user-agent"]:
        value = request.headers.get(header_name)
        if value:
            headers[header_name] = value
    headers["content-type"] = "application/json"
    return headers


async def parse_body(request: Request) -> Dict[str, Any]:
    body = await request.body()
    return json.loads(body.decode("utf-8")) if body else {}


@router.post("/v1/messages")
async def create_message(request_model: MessagesRequest, raw_request: Request):
    require_bearer_api_key(raw_request)
    raw_body = await parse_body(raw_request)
    normalized_body = maybe_remap_model(request_model.model_dump(exclude_none=True), enable_model_remap=settings.enable_model_remap)
    outgoing_body = merge_request_body(raw_body, normalized_body, forward_unknown_fields=settings.forward_unknown_fields)
    forward_headers = build_forward_headers(raw_request)

    if settings.log_request_body:
        print(json.dumps(outgoing_body, ensure_ascii=False))

    if outgoing_body.get("stream"):
        upstream_response = await client.post_stream("/api/v1/messages", headers=forward_headers, json_body=outgoing_body)
        if upstream_response.status_code >= 400:
            logger.error("Upstream stream request failed: status=%s body=%s", upstream_response.status_code, await upstream_response.aread())
        return streaming_response(upstream_response)

    upstream_response = await client.post_json("/api/v1/messages", headers=forward_headers, json_body=outgoing_body)
    if upstream_response.status_code >= 400:
        logger.error("Upstream request failed: status=%s body=%s", upstream_response.status_code, upstream_response.text)
    return JSONResponse(content=upstream_response.json(), status_code=upstream_response.status_code, headers={k: v for k, v in upstream_response.headers.items() if k.lower() in {"content-type", "x-request-id"}})


@router.post("/v1/messages/count_tokens")
async def count_tokens(request_model: TokenCountRequest, raw_request: Request):
    require_bearer_api_key(raw_request)
    raw_body = await parse_body(raw_request)
    normalized_body = maybe_remap_model(request_model.model_dump(exclude_none=True), enable_model_remap=settings.enable_model_remap)
    outgoing_body = merge_request_body(raw_body, normalized_body, forward_unknown_fields=settings.forward_unknown_fields)
    forward_headers = build_forward_headers(raw_request)

    upstream_response = await client.post_json("/api/v1/messages/count_tokens", headers=forward_headers, json_body=outgoing_body)
    return JSONResponse(content=upstream_response.json(), status_code=upstream_response.status_code, headers={k: v for k, v in upstream_response.headers.items() if k.lower() in {"content-type", "x-request-id"}})


@router.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Anthropic to CRS proxy"}


@router.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}
