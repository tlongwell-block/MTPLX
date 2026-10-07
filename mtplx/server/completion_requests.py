"""OpenAI completion request shapes shared by HTTP serving backends."""

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = Field(
        default=None, validation_alias=AliasChoices("top_p", "topP")
    )
    top_k: int | None = Field(
        default=None, validation_alias=AliasChoices("top_k", "topK")
    )
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    depth: int | None = None
    draft_block_size: int | None = None
    gemma_draft_block_size: int | None = None
    generation_mode: str | None = None
    seed: int | None = None
    enable_thinking: bool | None = None
    reasoning_effort: str | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    stop: Any = None
    stream_options: dict[str, Any] | None = None
    response_format: Any = None
    metadata: dict[str, Any] | None = None
    user: str | None = None
    # Declared so a logprobs request fails loudly (400) instead of being
    # silently swallowed by extra="allow" — clients were reading absent
    # logprobs as "model returned none" rather than "server ignored me".
    logprobs: Any = None
    top_logprobs: int | None = None


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    prompt: str | list[int] | list[str] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    depth: int | None = None
    draft_block_size: int | None = None
    gemma_draft_block_size: int | None = None
    generation_mode: str | None = None
    seed: int | None = None
    stop: Any = None
    stream: bool = False
    # Prompt scoring (echo + logprobs + max_tokens 0): one teacher-forced
    # pass returning per-position top-K logprobs — the lane KL-divergence
    # harnesses consume. Decode-time logprobs remain unsupported.
    echo: bool = False
    logprobs: int | None = None


