"""LLM client abstraction for the text-to-SQL assistant.

A tiny Protocol (`LLMClient`) decouples the agent from any specific LLM SDK, so:
  - production uses `GroqClient` (reads GROQ_API_KEY from env);
  - tests inject a scripted fake and never touch the network or need a key.

Provider: **Groq free tier** serving `openai/gpt-oss-120b` (see DECISIONS.md for
why, and why not Llama). The whole agent loop / validator / executor only depends
on `complete(system, user)`, so swapping providers is a one-file change — which is
exactly how this codebase moved off Anthropic without touching anything downstream.
"""
from __future__ import annotations

import os
from typing import Protocol

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.environ.get("TEXT2SQL_MODEL", "openai/gpt-oss-120b")


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str:  # pragma: no cover - interface
        ...


class GroqClient:
    """Thin wrapper over Groq's OpenAI-compatible chat completions API."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1024):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
                "(no credit card required) and add it to .env "
                "(needed only for the /ask text-to-SQL endpoint)."
            )
        import groq

        self._client = groq.Groq(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


def default_client() -> LLMClient:
    return GroqClient()
