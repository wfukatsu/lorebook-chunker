"""OpenAI placeholder: コンストラクタで即失敗して fail-fast する."""
from __future__ import annotations

from lorebook_chunker.llm import GenerateResult, LLMPermanentError


class OpenAIPlaceholderClient:
    """v1 では実装しない. `--llm-backend openai` が選ばれた時点で失敗させる.

    ingest のチャンク化等に工数を払う前に CLI 起動段階で fail する設計 (plan の Key Decisions 参照).
    """

    model_id = "openai@not-implemented"

    def __init__(self, **_: object) -> None:
        raise LLMPermanentError(
            "OpenAI backend is a v2 feature — use --llm-backend anthropic or ollama"
        )

    def generate(self, prompt: str, max_tokens: int) -> GenerateResult:  # pragma: no cover
        raise LLMPermanentError("OpenAI backend is a v2 feature")
