"""Anthropic Claude SDK adapter."""
from __future__ import annotations

import os
from typing import Any

from lorebook_chunker.llm import (
    GenerateResult,
    LLMPermanentError,
    LLMRetryableError,
)

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_SYSTEM_PROMPT = "あなたは日本語の正確な要約を書くアシスタントです。"


class AnthropicLLMClient:
    """Anthropic API を使った LLMClient 実装."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        import anthropic

        self.model_id = model
        self._system_prompt = system_prompt
        self._client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"), timeout=timeout)

    def generate(self, prompt: str, max_tokens: int) -> GenerateResult:
        import anthropic

        try:
            response = self._client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                system=self._system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.RateLimitError as e:
            raise LLMRetryableError(f"rate limited: {e}") from e
        except anthropic.APITimeoutError as e:
            raise LLMRetryableError(f"timeout: {e}") from e
        except anthropic.APIConnectionError as e:
            raise LLMRetryableError(f"connection error: {e}") from e
        except anthropic.AuthenticationError as e:
            raise LLMPermanentError(f"authentication failed: {e}") from e
        except anthropic.PermissionDeniedError as e:
            raise LLMPermanentError(f"permission denied: {e}") from e
        except anthropic.NotFoundError as e:
            raise LLMPermanentError(f"model or resource not found: {e}") from e
        except anthropic.APIStatusError as e:
            if 500 <= e.status_code < 600:
                raise LLMRetryableError(f"server error {e.status_code}: {e}") from e
            raise LLMPermanentError(f"API status {e.status_code}: {e}") from e
        return _normalize_response(response)


def _normalize_response(response: Any) -> GenerateResult:
    text_parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(getattr(block, "text", "") or "")
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    finish = str(getattr(response, "stop_reason", "") or "")
    # Anthropic 値を正規化: "end_turn" はそのまま、それ以外もそのまま通す (wiki 側で判定)
    return GenerateResult(
        text="".join(text_parts),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_id=str(getattr(response, "model", "") or ""),
        finish_reason=finish or "unknown",
    )
