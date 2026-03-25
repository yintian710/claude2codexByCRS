from __future__ import annotations

from typing import Any, Dict


EFFORT_MAP = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "extra high": "xhigh",
    "xhigh": "xhigh",
}


OPENAI_CODEX_INSTRUCTIONS = """You are Codex, based on GPT-5. You are running as a coding agent in the Codex CLI on a user's computer.

## General

- When searching for text or files, prefer using `rg` or `rg --files` respectively because `rg` is much faster than alternatives like `grep`. (If the `rg` command is not found, then use alternatives.)

## Editing constraints

- Default to ASCII when editing or creating files. Only introduce non-ASCII or other Unicode characters when there is a clear justification and the file already uses them.
- Add succinct code comments that explain what is going on if code is not self-explanatory. Usage of these comments should be rare.
- Try to use apply_patch for single file edits, but it is fine to explore other options to make the edit if it does not work well.
- Do not amend a commit unless explicitly requested to do so.
- NEVER use destructive commands like `git reset --hard` or `git checkout --` unless specifically requested or approved by the user.

## Plan tool

- Skip using the planning tool for straightforward tasks.
- Do not make single-step plans.

## Tone and style

- Be concise.
- Focus on getting the task done end to end.
"""


def map_effort_value(raw_effort: str | None) -> str:
    normalized_effort = (raw_effort or "medium").strip().lower().replace("-", " ").replace("_", " ")
    return EFFORT_MAP.get(normalized_effort, "medium")


def get_reasoning_effort(body_json: Dict[str, Any]) -> str:
    output_config = body_json.get("output_config")
    raw_effort = output_config.get("effort") if isinstance(output_config, dict) else None
    return map_effort_value(raw_effort)


def convert_tool_choice(tool_choice: Dict[str, Any] | None) -> Any:
    if not tool_choice:
        return None

    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "tool" and tool_choice.get("name"):
        return {
            "type": "function",
            "function": {"name": tool_choice["name"]},
        }
    return "auto"


def convert_tools(tools: list[Dict[str, Any]] | None) -> list[Dict[str, Any]] | None:
    if not tools:
        return None

    converted_tools = []
    for tool in tools:
        converted_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
        )
    return converted_tools
