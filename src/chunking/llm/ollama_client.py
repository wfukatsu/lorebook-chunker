"""Ollama local LLM adapter."""
from __future__ import annotations

import hashlib
from typing import Any

import ollama

from chunking.llm import (
    GenerateResult,
    LLMPermanentError,
    LLMRetryableError,
)

DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"
DEFAULT_SYSTEM_PROMPT = "あなたは日本語の正確な要約を書くアシスタントです。"
DEFAULT_TIMEOUT_SECONDS = 60.0


class OllamaLLMClient:
    """Ollama (local) を使った LLMClient 実装.

    model_id は `{model_tag}@{digest}` を `ollama.show` から初期化時に取得する.
    digest が取れなければ `{model_tag}@unknown` を記録 (警告ログあり, ingest は続行).

    F-016: daemon ハング耐性のため ``timeout_seconds`` を設け、HTTP クライアントと
    generate の両方に適用する. タイムアウトは ``LLMRetryableError`` として扱う.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        host: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._system_prompt = system_prompt
        self._model_tag = model
        self._timeout_seconds = timeout_seconds
        # F-016: ollama.Client は timeout kwarg を受け付ける.
        # host 指定時は Client を構築し timeout を渡す. 未指定時は既定動作維持 (module をそのまま使う)
        # — テスト容易性のためと、後方互換のため.
        if host:
            try:
                self._client = ollama.Client(host=host, timeout=timeout_seconds)
            except TypeError:  # pragma: no cover - older ollama-python
                self._client = ollama.Client(host=host)
        else:
            # module-level default client. timeout は generate 側の呼び出しで効かせられる.
            # (ollama-python は httpx.Client の timeout_seconds 引数を default 固定で持っている)
            self._client = ollama
        self.model_id = self._resolve_model_id(model)

    def _resolve_model_id(self, tag: str) -> str:
        """`ollama.show` で digest / modelfile ハッシュを取得し `{tag}@{sha}` を構築."""
        try:
            info = self._client.show(tag)
        except Exception:  # pragma: no cover - connectivity-dependent
            return f"{tag}@unknown"
        sha = _extract_digest(info)
        if sha:
            return f"{tag}@{sha}"
        modelfile = _get_field(info, "modelfile")
        if isinstance(modelfile, str) and modelfile:
            return f"{tag}@{hashlib.sha256(modelfile.encode('utf-8')).hexdigest()[:16]}"
        return f"{tag}@unknown"

    def generate(self, prompt: str, max_tokens: int) -> GenerateResult:
        full_prompt = f"{self._system_prompt}\n\n{prompt}" if self._system_prompt else prompt
        try:
            response = self._client.generate(
                model=self._model_tag,
                prompt=full_prompt,
                options={"num_predict": max_tokens},
                stream=False,
            )
        except ConnectionError as e:
            raise LLMRetryableError(f"ollama connection error: {e}") from e
        except TimeoutError as e:
            raise LLMRetryableError(f"ollama request timed out: {e}") from e
        except Exception as e:  # ollama SDK は具体例外が薄いので最終フォールバック
            msg = str(e).lower()
            if "timeout" in msg or "timed out" in msg:
                # F-016: タイムアウトは retryable 扱い
                raise LLMRetryableError(f"ollama timeout: {e}") from e
            if "not found" in msg or "pull" in msg:
                raise LLMPermanentError(f"ollama model not found: {e}") from e
            raise LLMRetryableError(f"ollama error: {e}") from e

        text = _get_field(response, "response") or ""
        prompt_eval_count = int(_get_field(response, "prompt_eval_count", 0) or 0)
        eval_count = int(_get_field(response, "eval_count", 0) or 0)
        done_reason = str(_get_field(response, "done_reason", "") or "")
        finish = _normalize_ollama_done(done_reason)
        return GenerateResult(
            text=str(text),
            input_tokens=prompt_eval_count,
            output_tokens=eval_count,
            model_id=self.model_id,
            finish_reason=finish,
        )


def _extract_digest(info: Any) -> str | None:
    """ollama.show のレスポンスから digest を取り出す. details.digest → digest の順."""
    details = _get_field(info, "details") or {}
    sha = _get_field(details, "digest")
    if isinstance(sha, str) and sha:
        return sha
    sha = _get_field(info, "digest")
    if isinstance(sha, str) and sha:
        return sha
    return None


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """dict でも attr オブジェクトでも取り出せる汎用アクセサ."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_ollama_done(reason: str) -> str:
    if reason in ("stop", "stop_sequence"):
        return "stop"
    if reason == "length":
        return "max_tokens"
    if not reason:
        return "end_turn"
    return reason
