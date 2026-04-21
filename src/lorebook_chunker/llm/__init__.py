"""LLM client abstraction.

Protocol + GenerateResult + error hierarchy + factory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class GenerateResult:
    """1 回の LLM 呼び出しの結果.

    text: 生成テキスト
    input_tokens / output_tokens: 課金/計測用トークン数
    model_id: 生成に使ったモデル識別子 (source_hash 計算で使う。Ollama なら `qwen2.5@sha256...`)
    finish_reason: "end_turn" / "stop" / "max_tokens" / "safety" / "error" 等、プロバイダ正規化後の値
    """

    text: str
    input_tokens: int
    output_tokens: int
    model_id: str
    finish_reason: str


class LLMError(Exception):
    """LLM 呼び出し関連の最上位例外."""


class LLMRetryableError(LLMError):
    """リトライすれば成功する可能性のある一時エラー (rate limit / timeout / transient 5xx)."""


class LLMPermanentError(LLMError):
    """リトライしても回復しない恒久エラー (auth / permission / not_found / v2 未実装)."""


class LLMPreflightError(LLMPermanentError):
    """pre-flight 呼び出しが成功条件を満たさなかった (ingest を即時 abort させる)."""


class LLMClient(Protocol):
    """全バックエンドが実装すべき最小インターフェース."""

    model_id: str

    def generate(self, prompt: str, max_tokens: int) -> GenerateResult: ...


def get_client(backend: str, config: dict[str, Any] | None = None) -> LLMClient:
    """設定から LLMClient を生成. OpenAI は placeholder で即 LLMPermanentError.

    config example:
      {"model": "claude-haiku-4-5", "system_prompt": "..."}
    """
    config = config or {}
    if backend == "anthropic":
        from lorebook_chunker.llm.anthropic_client import AnthropicLLMClient

        return AnthropicLLMClient(**config)
    if backend == "ollama":
        from lorebook_chunker.llm.ollama_client import OllamaLLMClient

        return OllamaLLMClient(**config)
    if backend == "openai":
        from lorebook_chunker.llm.openai_client import OpenAIPlaceholderClient

        return OpenAIPlaceholderClient(**config)
    raise LLMPermanentError(f"unknown LLM backend: {backend!r}")


__all__ = [
    "GenerateResult",
    "LLMClient",
    "LLMError",
    "LLMRetryableError",
    "LLMPermanentError",
    "LLMPreflightError",
    "get_client",
]
