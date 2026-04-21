"""LLM client abstraction のテスト. 全て mock/factory ベースで API コールは行わない."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from chunking.llm import (
    GenerateResult,
    LLMPermanentError,
    LLMPreflightError,
    LLMRetryableError,
    get_client,
)


# ---- Factory tests ------------------------------------------------------

def test_openai_factory_fails_immediately() -> None:
    with pytest.raises(LLMPermanentError) as exc_info:
        get_client("openai", {})
    assert "v2" in str(exc_info.value)


def test_unknown_backend_raises() -> None:
    with pytest.raises(LLMPermanentError):
        get_client("gemini", {})


# ---- Anthropic client ---------------------------------------------------

def _anthropic_mock_response(text: str, finish: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=42, output_tokens=13),
        stop_reason=finish,
        model="claude-haiku-4-5",
    )


def test_anthropic_client_normalizes_response(monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic

    from chunking.llm.anthropic_client import AnthropicLLMClient

    def fake_create(**kwargs):
        assert kwargs["model"] == "claude-haiku-4-5"
        return _anthropic_mock_response("要約本文")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    client = AnthropicLLMClient()
    monkeypatch.setattr(client._client.messages, "create", fake_create)
    result = client.generate("テスト", max_tokens=100)
    assert isinstance(result, GenerateResult)
    assert result.text == "要約本文"
    assert result.input_tokens == 42
    assert result.output_tokens == 13
    assert result.finish_reason == "end_turn"


def test_anthropic_rate_limit_becomes_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic
    from chunking.llm.anthropic_client import AnthropicLLMClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    client = AnthropicLLMClient()

    def fake_create(**_):
        raise anthropic.RateLimitError(
            message="rate limit",
            response=SimpleNamespace(status_code=429, headers={}, request=None),
            body=None,
        )

    monkeypatch.setattr(client._client.messages, "create", fake_create)
    with pytest.raises(LLMRetryableError):
        client.generate("x", max_tokens=10)


def test_anthropic_auth_error_becomes_permanent(monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic
    from chunking.llm.anthropic_client import AnthropicLLMClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    client = AnthropicLLMClient()

    def fake_create(**_):
        raise anthropic.AuthenticationError(
            message="bad key",
            response=SimpleNamespace(status_code=401, headers={}, request=None),
            body=None,
        )

    monkeypatch.setattr(client._client.messages, "create", fake_create)
    with pytest.raises(LLMPermanentError):
        client.generate("x", max_tokens=10)


# ---- Ollama client ------------------------------------------------------

def test_ollama_client_uses_digest_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import chunking.llm.ollama_client as mod

    class _FakeClient:
        def show(self, tag: str) -> dict:
            return {"details": {"digest": "sha256abcdef"}}

        def generate(self, **kwargs) -> dict:
            return {
                "response": "要約テキスト",
                "prompt_eval_count": 10,
                "eval_count": 20,
                "done_reason": "stop",
            }

    fake = _FakeClient()
    # host=None のケースで `ollama` モジュール自体が使われる
    monkeypatch.setattr(mod, "ollama", fake)
    client = mod.OllamaLLMClient(model="qwen2.5:7b")
    assert client.model_id == "qwen2.5:7b@sha256abcdef"
    result = client.generate("x", max_tokens=50)
    assert result.text == "要約テキスト"
    assert result.input_tokens == 10
    assert result.output_tokens == 20
    assert result.finish_reason == "stop"
    assert result.model_id == "qwen2.5:7b@sha256abcdef"


def test_ollama_client_passes_think_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen3 系は think=False を渡さないと response が空になる (F-016 拡張)."""
    import chunking.llm.ollama_client as mod

    captured: dict = {}

    class _FakeClient:
        def show(self, tag: str) -> dict:
            return {"details": {"digest": "sha"}}

        def generate(self, **kwargs) -> dict:
            captured.update(kwargs)
            return {"response": "ok", "eval_count": 1, "done_reason": "stop"}

    monkeypatch.setattr(mod, "ollama", _FakeClient())
    mod.OllamaLLMClient(model="qwen3:8b").generate("x", max_tokens=10)
    assert captured.get("think") is False

    captured.clear()
    mod.OllamaLLMClient(model="qwen3:8b", think=True).generate("x", max_tokens=10)
    assert captured.get("think") is True


def test_ollama_client_falls_back_when_think_kwarg_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ollama-python <0.4 では think kwarg 未対応 → TypeError fallback."""
    import chunking.llm.ollama_client as mod

    call_log: list[dict] = []

    class _FakeClient:
        def show(self, tag: str) -> dict:
            return {"details": {"digest": "sha"}}

        def generate(self, **kwargs) -> dict:
            call_log.append(kwargs)
            if "think" in kwargs:
                raise TypeError("generate() got an unexpected keyword argument 'think'")
            return {"response": "ok", "eval_count": 1, "done_reason": "stop"}

    monkeypatch.setattr(mod, "ollama", _FakeClient())
    result = mod.OllamaLLMClient(model="qwen3:8b").generate("x", max_tokens=10)
    assert result.text == "ok"
    assert len(call_log) == 2
    assert "think" in call_log[0]
    assert "think" not in call_log[1]


def test_ollama_client_falls_back_to_modelfile_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    import chunking.llm.ollama_client as mod

    class _FakeClient:
        def show(self, tag: str) -> dict:
            return {"modelfile": "FROM qwen\nPARAMETER temperature 0.2"}

        def generate(self, **kwargs):  # pragma: no cover - not hit
            raise AssertionError

    monkeypatch.setattr(mod, "ollama", _FakeClient())
    client = mod.OllamaLLMClient(model="qwen2.5:7b")
    assert client.model_id.startswith("qwen2.5:7b@")
    assert "unknown" not in client.model_id


def test_ollama_client_unknown_digest_marks_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    import chunking.llm.ollama_client as mod

    class _FakeClient:
        def show(self, tag: str) -> dict:
            return {}

        def generate(self, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(mod, "ollama", _FakeClient())
    client = mod.OllamaLLMClient(model="qwen2.5:7b")
    assert client.model_id == "qwen2.5:7b@unknown"


def test_ollama_generate_connection_error_is_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    import chunking.llm.ollama_client as mod

    class _FakeClient:
        def show(self, tag: str):
            return {"details": {"digest": "abc"}}

        def generate(self, **kwargs):
            raise ConnectionError("cant connect")

    monkeypatch.setattr(mod, "ollama", _FakeClient())
    client = mod.OllamaLLMClient(model="qwen2.5:7b")
    with pytest.raises(LLMRetryableError):
        client.generate("x", max_tokens=10)


def test_ollama_model_not_found_is_permanent(monkeypatch: pytest.MonkeyPatch) -> None:
    import chunking.llm.ollama_client as mod

    class _FakeClient:
        def show(self, tag: str):
            return {"details": {"digest": "abc"}}

        def generate(self, **kwargs):
            raise RuntimeError("model not found. pull it first.")

    monkeypatch.setattr(mod, "ollama", _FakeClient())
    client = mod.OllamaLLMClient(model="missing:latest")
    with pytest.raises(LLMPermanentError):
        client.generate("x", max_tokens=10)


# ---- Error hierarchy ---------------------------------------------------

def test_preflight_is_subclass_of_permanent() -> None:
    assert issubclass(LLMPreflightError, LLMPermanentError)
