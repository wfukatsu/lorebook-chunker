"""Query CLI テスト. stub analyzer で ingest を走らせ、query で取り出す."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from chunking.ingest import IngestConfig, IngestRunner
from chunking.llm import GenerateResult
from chunking.query import QueryResult, run_query, run_query_impl
from tests.test_ingest import _ScriptedLLM, _StubAnalyzer


def _run_ingest(tmp_path: Path, corpus_text: dict[str, str]) -> Path:
    """与えた corpus を samples/ に置き、ingest を走らせて out/ を作る."""
    input_dir = tmp_path / "samples"
    input_dir.mkdir()
    for name, content in corpus_text.items():
        (input_dir / name).write_text(content, encoding="utf-8")
    output_dir = tmp_path / "out"
    cfg = IngestConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        skip_wiki=True,
        target_chars=30,
        max_chunk_chars=200,
    )
    analyzer = _StubAnalyzer()
    IngestRunner(
        cfg,
        analyzer_factory=lambda path: analyzer,
        llm_factory=lambda backend, config: _ScriptedLLM([]),
    ).run()
    return output_dir


def test_query_matches_literal_term(tmp_path: Path) -> None:
    output_dir = _run_ingest(
        tmp_path,
        {
            "01.txt": "田中はスカラー商事に勤めている。"
            "田中は東京に住んでいる。"
            "スカラー商事は大阪に支社を持つ。",
            "02.txt": "佐藤は大阪に住んでいる。佐藤は別の会社で働いている。",
        },
    )
    analyzer = _StubAnalyzer()
    result = run_query_impl(
        output_dir, "田中とスカラー商事", top_k=3, analyzer_factory=lambda path: analyzer
    )
    assert result.exit_code == 0
    assert result.hits  # 少なくとも 1 件返る
    # 上位に「田中」「スカラー商事」が含まれるチャンクが来る
    top = result.hits[0]
    assert "田中" in top.text or "スカラー商事" in top.text


def test_query_on_missing_output_dir_returns_error(tmp_path: Path) -> None:
    result = run_query_impl(
        tmp_path / "nonexistent", "anything", top_k=5,
        analyzer_factory=lambda path: _StubAnalyzer(),
    )
    assert result.exit_code != 0
    assert "見つかりません" in result.error_message or "先に" in result.error_message


def test_query_on_incomplete_output_dir_returns_error(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "chunks.jsonl").write_text("", encoding="utf-8")  # vocab.npz 無し
    result = run_query_impl(
        out, "q", top_k=5,
        analyzer_factory=lambda path: _StubAnalyzer(),
    )
    assert result.exit_code != 0


def test_query_oov_returns_empty_hits_with_warning(tmp_path: Path) -> None:
    output_dir = _run_ingest(
        tmp_path,
        {
            "01.txt": "田中はスカラー商事に勤めている。"
            "田中は東京に住んでいる。",
        },
    )
    analyzer = _StubAnalyzer()
    result = run_query_impl(
        output_dir, "完全に別の語彙", top_k=3,
        analyzer_factory=lambda path: analyzer,
    )
    # F-038: OOV は exit_code=4 (正常なゼロマッチと区別する).
    assert result.exit_code == 4
    assert result.hits == []
    assert result.error_message


def test_query_respects_top_k(tmp_path: Path) -> None:
    output_dir = _run_ingest(
        tmp_path,
        {
            "01.txt": "田中はスカラー商事に勤めている。"
            "田中は東京に住んでいる。"
            "スカラー商事は大阪に支社を持つ。"
            "プロジェクトが進む。"
            "田中はプロジェクトを率いる。",
        },
    )
    analyzer = _StubAnalyzer()
    result = run_query_impl(
        output_dir, "田中", top_k=2,
        analyzer_factory=lambda path: analyzer,
    )
    assert len(result.hits) <= 2


def test_run_query_cli_wrapper_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    output_dir = _run_ingest(
        tmp_path,
        {
            "01.txt": "田中はスカラー商事に勤めている。田中は東京に住んでいる。",
        },
    )
    args = argparse.Namespace(
        output_dir=str(output_dir),
        query_text="田中",
        top_k=3,
        command="query",
    )
    import chunking.query as mod
    monkeypatch.setattr(
        "chunking.ingest.default_analyzer_factory", lambda path: _StubAnalyzer()
    )
    exit_code = run_query(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    # 何らかの結果が出ている
    assert "chunk_id" in out or "no matching" in out or "text" in out


def test_run_query_cli_wrapper_missing_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    args = argparse.Namespace(
        output_dir=str(tmp_path / "noop"),
        query_text="x",
        top_k=5,
        command="query",
    )
    exit_code = run_query(args)
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "error" in err
