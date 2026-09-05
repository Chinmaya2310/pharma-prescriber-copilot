"""LLM client abstraction for the text-to-SQL assistant.

A tiny Protocol (`LLMClient`) decouples the agent from the Anthropic SDK so that:
  - production uses `AnthropicClient` (reads ANTHROPIC_API_KEY from env);
  - tests inject a scripted fake and never touch the network or need a key.
"""
from __future__ import annotations

import os
from typing import Protocol

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.environ.get("TEXT2SQL_MODEL", "claude-opus-4-8")


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str:  # pragma: no cover - interface
        ...


class AnthropicClient:
    """Thin wrapper over the Anthropic Messages API."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1024):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key "
                "(needed only for the /ask text-to-SQL endpoint)."
            )
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


def default_client() -> LLMClient:
    return AnthropicClient()
