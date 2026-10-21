"""OpenRouter chat helper for course notebooks and experiments."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_SYSTEM = (
    "You are a patient AI-agent tutor."
    "Explain concepts simply and give Python examples."
)
DEFAULT_HEADERS = {
    "HTTP-Referer": "http://localhost",
    "X-OpenRouter-Title": "My AI Agents Learning Project",
}


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (Path.cwd(), *Path.cwd().parents):
        env_path = candidate / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=False)
            return


def get_client(
    api_key: str | None = None,
    timeout: float | None = None,
) -> OpenAI:
    """Return an OpenAI-compatible client pointed at OpenRouter."""
    _load_env()
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY before calling OpenRouter.")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, timeout=timeout)


def _build_messages(
    message: str | None,
    messages: list[dict[str, Any]] | None,
    system: str | None,
) -> list[dict[str, Any]]:
    if message is not None and messages is not None:
        raise ValueError("Pass either message= or messages=, not both.")
    if messages is not None:
        if not messages:
            raise ValueError("messages= cannot be empty.")
        return messages
    if message is None:
        raise ValueError("Pass message='...' or messages=[...].")

    built: list[dict[str, Any]] = []
    if system:
        built.append({"role": "system", "content": system})
    built.append({"role": "user", "content": message})
    return built


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if content:
        return content
    reasoning = getattr(message, "reasoning", None)
    return reasoning or ""


def _markdown_display(text: str = "", handle: Any | None = None) -> Any:
    """Show or update notebook Markdown. No-ops outside IPython."""
    try:
        from IPython.display import Markdown, display
    except ImportError:
        return None
    rendered = Markdown(text)
    if handle is not None:
        handle.update(rendered)
        return handle
    return display(rendered, display_id=True)


def _read_stream(
    stream: Any,
    echo: bool,
    markdown_handle: Any | None = None,
) -> tuple[str, Any]:
    pieces: list[str] = []
    usage = None
    for chunk in stream:
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None) or ""
        if text:
            pieces.append(text)
            if echo:
                print(text, end="", flush=True)
            if markdown_handle is not None:
                _markdown_display("".join(pieces), handle=markdown_handle)
    if echo:
        print()
    return "".join(pieces), usage


def openrouter_chat(
    message: str | None = None,
    *,
    messages: list[dict[str, Any]] | None = None,
    system: str | None = DEFAULT_SYSTEM,
    model: str = DEFAULT_MODEL,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stop: str | list[str] | None = None,
    seed: int | None = None,
    stream: bool = False,
    return_completion: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
    retries: int = 5,
    timeout: float | None = None,
    print_usage: bool = False,
    markdown: bool = False,
    echo_stream: bool | None = None,
    api_key: str | None = None,
    client: OpenAI | None = None,
) -> str | ChatCompletion:
    """Call OpenRouter and return assistant text.

    With ``markdown=True``, notebooks render the reply as Markdown.
    Streaming updates that same Markdown cell as tokens arrive.
    With ``return_completion=True``, return the full non-streaming completion.

    Short form::

        openrouter_chat("Explain what an AI agent is.")

    Full messages form::

        openrouter_chat(messages=[{"role": "user", "content": "Hi"}])
    """
    if return_completion and stream:
        raise ValueError("return_completion=True is not supported with stream=True.")

    request_messages = _build_messages(message, messages, system)
    openai_client = client or get_client(api_key=api_key, timeout=timeout)
    headers = {**DEFAULT_HEADERS, **(extra_headers or {})}

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": request_messages,
        "max_tokens": max_tokens,
        "stream": stream,
        "extra_headers": headers,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if stop is not None:
        kwargs["stop"] = stop
    if seed is not None:
        kwargs["seed"] = seed
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if response_format is not None:
        kwargs["response_format"] = response_format
    if extra_body:
        kwargs["extra_body"] = extra_body
    if stream:
        kwargs["stream_options"] = {"include_usage": True}

    echo = (not markdown) if echo_stream is None else echo_stream

    completion = None
    for attempt in range(retries):
        completion = openai_client.chat.completions.create(**kwargs)
        if stream:
            markdown_handle = _markdown_display("") if markdown else None
            text, usage = _read_stream(
                completion,
                echo=echo,
                markdown_handle=markdown_handle,
            )
            if text:
                if markdown_handle is not None:
                    _markdown_display(text, handle=markdown_handle)
                if print_usage:
                    print("Usage:", usage)
                return text
        elif getattr(completion, "choices", None):
            if return_completion:
                return completion
            text = _message_text(completion.choices[0].message)
            if markdown:
                _markdown_display(text)
            if print_usage:
                print("Usage:", completion.usage)
            return text

        error = getattr(completion, "error", None)
        print(
            f"Attempt {attempt + 1}: OpenRouter returned no choices ({error}). Retrying..."
        )
        time.sleep(2**attempt)

    dump = completion.model_dump() if completion is not None else None
    raise RuntimeError(
        "OpenRouter returned no choices. The free model is likely rate-limited "
        f"or unavailable. Last response: {dump}"
    )


OpenRouterChat = openrouter_chat
