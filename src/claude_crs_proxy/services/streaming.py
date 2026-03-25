from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any, Dict

from fastapi.responses import StreamingResponse


async def relay_sse(response) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_bytes():
            if chunk:
                yield chunk
    finally:
        client = getattr(response, "_client", None)
        await response.aclose()
        if client is not None:
            await client.aclose()


def streaming_response(response) -> StreamingResponse:
    media_type = response.headers.get("content-type", "text/event-stream")
    return StreamingResponse(relay_sse(response), media_type=media_type, status_code=response.status_code)


async def convert_openai_stream_to_anthropic(response, original_model: str) -> AsyncIterator[bytes]:
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    message_started = False
    text_block_started = False
    text_block_closed = False
    tool_states: Dict[int, Dict[str, Any]] = {}
    prompt_tokens = 0
    completion_tokens = 0
    stop_reason = "end_turn"
    buffer = ""

    def sse_event(name: str, data: Dict[str, Any]) -> bytes:
        return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()

    async def ensure_started() -> AsyncIterator[bytes]:
        nonlocal message_started, text_block_started
        if not message_started:
            message_started = True
            yield sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": original_model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "output_tokens": 0,
                        },
                    },
                },
            )
            yield sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            text_block_started = True

    async def close_text_block() -> AsyncIterator[bytes]:
        nonlocal text_block_closed
        if text_block_started and not text_block_closed:
            text_block_closed = True
            yield sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})

    async def finalize_stream() -> AsyncIterator[bytes]:
        for event in close_text_block():
            async for chunk in event:
                yield chunk
        for index in sorted(tool_states.keys()):
            if not tool_states[index].get("closed"):
                tool_states[index]["closed"] = True
                yield sse_event("content_block_stop", {"type": "content_block_stop", "index": index + 1})
        yield sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": completion_tokens},
            },
        )
        yield sse_event("message_stop", {"type": "message_stop"})
        yield b"data: [DONE]\n\n"

    try:
        async for raw_chunk in response.aiter_text():
            if not raw_chunk:
                continue
            buffer += raw_chunk.replace("\r\n", "\n")
            while "\n\n" in buffer:
                raw_event, buffer = buffer.split("\n\n", 1)
                if not raw_event.strip():
                    continue
                data_lines = [line[6:] for line in raw_event.split("\n") if line.startswith("data: ")]
                if not data_lines:
                    continue
                payload = "".join(data_lines)
                if payload == "[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                async for chunk in ensure_started():
                    yield chunk

                if event.get("usage"):
                    prompt_tokens = event["usage"].get("prompt_tokens", prompt_tokens)
                    completion_tokens = event["usage"].get("completion_tokens", completion_tokens)

                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}

                if delta.get("content"):
                    if text_block_closed:
                        continue
                    yield sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": delta["content"]},
                        },
                    )

                for tool_call in delta.get("tool_calls") or []:
                    tool_index = tool_call.get("index", 0)
                    state = tool_states.setdefault(
                        tool_index,
                        {
                            "started": False,
                            "id": tool_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                            "name": "",
                            "closed": False,
                        },
                    )
                    if tool_call.get("id"):
                        state["id"] = tool_call["id"]
                    function = tool_call.get("function") or {}
                    if function.get("name"):
                        state["name"] = function["name"]
                    if not state["started"]:
                        async for chunk in close_text_block():
                            yield chunk
                        state["started"] = True
                        yield sse_event(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": tool_index + 1,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": state["id"],
                                    "name": state["name"],
                                    "input": {},
                                },
                            },
                        )
                    if function.get("arguments"):
                        yield sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": tool_index + 1,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": function["arguments"],
                                },
                            },
                        )

                finish_reason = choice.get("finish_reason")
                if finish_reason == "tool_calls":
                    stop_reason = "tool_use"
                elif finish_reason == "length":
                    stop_reason = "max_tokens"
                elif finish_reason == "stop":
                    stop_reason = "end_turn"
    finally:
        client = getattr(response, "_client", None)
        await response.aclose()
        if client is not None:
            await client.aclose()

    async for chunk in finalize_stream():
        yield chunk


def anthropic_streaming_response(response, original_model: str) -> StreamingResponse:
    return StreamingResponse(
        convert_openai_stream_to_anthropic(response, original_model),
        media_type="text/event-stream",
        status_code=response.status_code,
    )