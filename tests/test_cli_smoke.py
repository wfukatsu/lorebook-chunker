"""CLI エントリポイントの最低限のスモークテスト."""
from __future__ import annotations

import pytest

from chunking.cli import IDENTITY_BANNER, build_parser


def test_identity_banner_is_nonempty() -> None:
    assert IDENTITY_BANNER
    assert "RAG" in IDENTITY_BANNER


def test_parser_has_three_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    out = capsys.readouterr().out
    assert "ingest" in out
    assert "query" in out
    assert "lint" in out


def test_query_requires_top_k_integer() -> None:
    parser = build_parser()
    args = parser.parse_args(["query", "out", "foo"])
    assert args.top_k == 5
    args = parser.parse_args(["query", "out", "foo", "--top-k", "10"])
    assert args.top_k == 10


def test_ingest_default_backend_is_anthropic() -> None:
    parser = build_parser()
    args = parser.parse_args(["ingest", "in", "out"])
    assert args.llm_backend == "anthropic"
    assert args.llm_model is None


def test_ingest_accepts_llm_model_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["ingest", "in", "out", "--llm-backend", "ollama", "--llm-model", "qwen3:8b"]
    )
    assert args.llm_backend == "ollama"
    assert args.llm_model == "qwen3:8b"
