from __future__ import annotations

import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from litellm import token_counter

logger = logging.getLogger(__name__)

from src.claude_crs_proxy.config import settings
from src.claude_crs_proxy.schemas import MessagesRequest, TokenCountRequest
from src.claude_crs_proxy.services.auth import require_bearer_api_key
from src.claude_crs_proxy.services.converter import (
    convert_anthropic_to_openai_chat,
    convert_openai_to_anthropic,
)
from src.claude_crs_proxy.services.crs_client import CRSClient
from src.claude_crs_proxy.services.model_routing import maybe_remap_model
from src.claude_crs_proxy.services.streaming import anthropic_streaming_response

router = APIRouter()
client = CRSClient()


def build_forward_headers(request: Request) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for header_name in [
        "authorization",
        "x-request-id",
        "x-stainless-arch",
        "x-stainless-lang",
        "x-stainless-os",
        "x-stainless-package-version",
        "x-stainless-runtime",
        "x-stainless-runtime-version",
        "x-stainless-timeout",
        "x-app",
        "user-agent",
    ]:
        value = request.headers.get(header_name)
        if value:
            headers[header_name] = value
    headers["content-type"] = "application/json"
    return headers


async def parse_body(request: Request) -> Dict[str, Any]:
    body = await request.body()
    return json.loads(body.decode("utf-8")) if body else {}


def build_json_response(upstream_response) -> JSONResponse:
    response_headers = {
        key: value
        for key, value in upstream_response.headers.items()
        if key.lower() in {"content-type", "x-request-id"}
    }
    return JSONResponse(
        content=upstream_response.json(),
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


@router.post("/v1/messages")
async def create_message(request_model: MessagesRequest, raw_request: Request):
    require_bearer_api_key(raw_request)
    raw_body = await parse_body(raw_request)
    mapped_model = maybe_remap_model(request_model.model, enable_model_remap=settings.enable_model_remap)
    outgoing_body = convert_anthropic_to_openai_chat(request_model, raw_body, mapped_model)
    forward_headers = build_forward_headers(raw_request)

    if settings.log_request_body:
        print(json.dumps(outgoing_body, ensure_ascii=False))

    if outgoing_body.get("stream"):
        upstream_response = await client.post_stream(
            "/v1/chat/completions",
            headers=forward_headers,
            json_body=outgoing_body,
        )
        if upstream_response.status_code >= 400:
            logger.error(
                "Upstream stream request failed: status=%s body=%s",
                upstream_response.status_code,
                await upstream_response.aread(),
            )
        return anthropic_streaming_response(upstream_response, request_model.model)

    upstream_response = await client.post_json(
        "/v1/chat/completions",
        headers=forward_headers,
        json_body=outgoing_body,
    )
    if upstream_response.status_code >= 400:
        logger.error(
            "Upstream request failed: status=%s body=%s",
            upstream_response.status_code,
            upstream_response.text,
        )
        return build_json_response(upstream_response)

    anthropic_response = convert_openai_to_anthropic(upstream_response.json(), request_model.model)
    return JSONResponse(content=anthropic_response, status_code=upstream_response.status_code)


@router.post("/v1/messages/count_tokens")
async def count_tokens(request_model: TokenCountRequest, raw_request: Request):
    require_bearer_api_key(raw_request)
    raw_body = await parse_body(raw_request)
    mapped_model = maybe_remap_model(request_model.model, enable_model_remap=settings.enable_model_remap)
    synthetic_request = MessagesRequest(
        model=request_model.model,
        max_tokens=raw_body.get("max_tokens", 1024),
        messages=request_model.messages,
        system=request_model.system,
        stop_sequences=raw_body.get("stop_sequences"),
        stream=bool(raw_body.get("stream", False)),
        temperature=raw_body.get("temperature", 1.0),
        top_p=raw_body.get("top_p"),
        top_k=raw_body.get("top_k"),
        metadata=request_model.metadata if hasattr(request_model, "metadata") else raw_body.get("metadata"),
        tools=request_model.tools,
        tool_choice=request_model.tool_choice,
        thinking=request_model.thinking,
        output_config=request_model.output_config,
        reasoning=request_model.reasoning,
        context_management=request_model.context_management,
        container=request_model.container,
        service_tier=request_model.service_tier,
    )
    chat_body = convert_anthropic_to_openai_chat(synthetic_request, raw_body, mapped_model)
    token_count = token_counter(model=chat_body["model"], messages=chat_body["messages"])
    return JSONResponse(content={"input_tokens": token_count}, status_code=200)


@router.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Anthropic to CRS Codex proxy"}


@router.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}