"""/ask endpoint — the agentic text-to-SQL assistant."""
from __future__ import annotations

import anthropic
from fastapi import APIRouter, HTTPException

from src.api.schemas import AskRequest, AskResponse
from src.text2sql.agent_loop import ask as run_ask
from src.text2sql.llm import AnthropicClient

router = APIRouter(tags=["assistant"])


@router.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    """Answer a plain-English question by generating + running read-only SQL.

    Returns the natural-language answer plus the SQL used, the raw result rows,
    and the full attempt trace, so every answer is auditable and grounded.
    """
    try:
        client = AnthropicClient()
    except RuntimeError as exc:
        # No API key configured — fail clearly instead of pretending.
        raise HTTPException(503, str(exc)) from exc

    try:
        result = run_ask(req.question, client=client, max_attempts=req.max_attempts)
    except anthropic.AuthenticationError as exc:
        raise HTTPException(502, "Anthropic auth failed — the ANTHROPIC_API_KEY is "
                            "invalid or revoked.") from exc
    except anthropic.APIError as exc:
        raise HTTPException(502, f"Anthropic API error: {exc}") from exc
    return AskResponse(**result.to_dict())
