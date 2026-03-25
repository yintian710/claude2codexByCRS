from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
                ContentBlockImage,
                ContentBlockToolUse,
                ContentBlockToolResult,
            ]
        ],
    ]


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ThinkingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")


class MessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[Union[ThinkingConfig, Dict[str, Any]]] = None
    output_config: Optional[Dict[str, Any]] = None
    reasoning: Optional[Dict[str, Any]] = None
    context_management: Optional[Dict[str, Any]] = None
    container: Optional[Dict[str, Any]] = None
    service_tier: Optional[str] = None


class TokenCountRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[Union[ThinkingConfig, Dict[str, Any]]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    output_config: Optional[Dict[str, Any]] = None
    reasoning: Optional[Dict[str, Any]] = None
    context_management: Optional[Dict[str, Any]] = None
    container: Optional[Dict[str, Any]] = None
    service_tier: Optional[str] = None


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse, Dict[str, Any]]]
    type: Literal["message"] = "message"
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Usage
