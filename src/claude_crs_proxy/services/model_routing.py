from __future__ import annotations

from src.claude_crs_proxy.config import settings


def map_model_name(model_name: str) -> str:
    clean_model = model_name.strip()

    if clean_model.startswith(("openai/", "anthropic/", "gemini/")):
        clean_model = clean_model.split("/", 1)[1]

    lowered = clean_model.lower()
    if "haiku" in lowered:
        return settings.small_model
    if "sonnet" in lowered:
        return settings.big_model
    if clean_model.startswith("gpt-"):
        return clean_model

    return clean_model


def maybe_remap_model(model_name: str, *, enable_model_remap: bool) -> str:
    if not enable_model_remap:
        return model_name
    return map_model_name(model_name)