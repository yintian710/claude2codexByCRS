from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List

from src.claude_crs_proxy.schemas import MessagesRequest
from src.claude_crs_proxy.services.field_mapper import (
    OPENAI_CODEX_INSTRUCTIONS,
    convert_tool_choice,
    convert_tools,
    get_reasoning_effort,
)


def _normalize_system_text(system: Any) -> str | None:
    if not system:
        return None
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: List[str] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif hasattr(block, "type") and getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        joined = "\n\n".join(part for part in parts if part)
        return joined or None
    return str(system)


def _stringify_tool_result_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(item.get("text", ""))
            elif isinstance(item, dict) and "text" in item:
                chunks.append(str(item.get("text", "")))
            else:
                try:
                    chunks.append(json.dumps(item, ensure_ascii=False))
                except TypeError:
                    chunks.append(str(item))
        return "\n".join(chunk for chunk in chunks if chunk)
    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text", ""))
        try:
            return json.dumps(content, ensure_ascii=False)
        except TypeError:
            return str(content)
    return str(content)


def _convert_message_content(content: Any, role: str) -> tuple[str | None, list[Dict[str, Any]] | None]:
    if isinstance(content, str):
        return content, None

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for block in content or []:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type == "text":
            text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
            if text:
                text_parts.append(text)
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id") if isinstance(block, dict) else getattr(block, "id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name") if isinstance(block, dict) else getattr(block, "name"),
                        "arguments": json.dumps(
                            block.get("input", {}) if isinstance(block, dict) else getattr(block, "input", {}),
                            ensure_ascii=False,
                        ),
                    },
                }
            )
        elif block_type == "tool_result" and role == "user":
            tool_result = _stringify_tool_result_content(
                block.get("content") if isinstance(block, dict) else getattr(block, "content", None)
            )
            text_parts.append(tool_result)
        elif block_type == "image":
            text_parts.append("[Image content omitted]")

    text = "\n".join(part for part in text_parts if part).strip() or None
    return text, tool_calls or None


def convert_anthropic_to_openai_chat(request: MessagesRequest, raw_body: Dict[str, Any], mapped_model: str) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []

    system_text = _normalize_system_text(raw_body.get("system", request.system))
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for message in request.messages:
        text_content, tool_calls = _convert_message_content(message.content, message.role)
        converted: Dict[str, Any] = {"role": message.role}
        if text_content is not None:
            converted["content"] = text_content
        else:
            converted["content"] = ""
        if tool_calls:
            converted["tool_calls"] = tool_calls
        messages.append(converted)

    chat_body: Dict[str, Any] = {
        "model": mapped_model,
        "messages": messages,
        "stream": bool(raw_body.get("stream", request.stream)),
        "reasoning_effort": get_reasoning_effort(raw_body),
        "instructions": OPENAI_CODEX_INSTRUCTIONS,
    }

    if request.temperature is not None:
        chat_body["temperature"] = request.temperature
    if request.top_p is not None:
        chat_body["top_p"] = request.top_p
    if request.metadata is not None:
        chat_body["metadata"] = request.metadata

    tools = convert_tools(raw_body.get("tools"))
    if tools:
        chat_body["tools"] = tools

    tool_choice = convert_tool_choice(raw_body.get("tool_choice"))
    if tool_choice is not None:
        chat_body["tool_choice"] = tool_choice

    return chat_body


def convert_openai_to_anthropic(response_data: Dict[str, Any], original_model: str) -> Dict[str, Any]:
    choices = response_data.get("choices") or [{}]
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content: List[Dict[str, Any]] = []

    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})

    for tool_call in message.get("tool_calls") or []:
        arguments = tool_call.get("function", {}).get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": tool_call.get("function", {}).get("name", ""),
                "input": arguments,
            }
        )

    if not content:
        content.append({"type": "text", "text": ""})

    finish_reason = choice.get("finish_reason")
    stop_reason = "end_turn"
    if finish_reason == "length":
        stop_reason = "max_tokens"
    elif finish_reason == "tool_calls":
        stop_reason = "tool_use"

    usage = response_data.get("usage") or {}

    return {
        "id": response_data.get("id", f"msg_{uuid.uuid4()}"),
        "type": "message",
        "role": "assistant",
        "model": original_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }
