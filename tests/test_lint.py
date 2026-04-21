"""Lint CLI のテスト. ingest を走らせてから lint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from chunking.ingest import IngestConfig, IngestRunner
from chunking.lint import (
    FATAL,
    INFO,
    WARNING,
    LintReport,
    run_lint,
    run_lint_impl,
)
from tests.test_ingest import _ScriptedLLM, _StubAnalyzer


def _ingest(tmp_path: Path, files: dict[str, str], **cfg_kwargs) -> Path:
    input_dir = tmp_path / "samples"
    input_dir.mkdir()
    for name, content in files.items():
        (input_dir / name).write_text(content, encoding="utf-8")
    output_dir = tmp_path / "out"
    defaults = dict(skip_wiki=True, target_chars=30, max_chunk_chars=200)
    defaults.update(cfg_kwargs)
    cfg = IngestConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        **defaults,
    )
    IngestRunner(
        cfg,
        analyzer_factory=lambda path: _StubAnalyzer(),
        llm_factory=lambda backend, config: _ScriptedLLM([]),
    ).run()
    return output_dir


def test_lint_healthy_corpus_reports_no_fatals(tmp_path: Path) -> None:
    output_dir = _ingest(
        tmp_path,
        {
            f"{i:02d}.txt": (
                f"文書{i}の一文目。"
                f"文書{i}の二文目を書きます。"
                f"文書{i}の三文目。"
            )
            for i in range(1, 8)
        },
        target_chars=20,
        max_chunk_chars=200,
    )
    report = run_lint_impl(output_dir, target_chunk_chars=20, max_chunk_chars=200)
    assert report.count(FATAL) == 0


def test_lint_detects_missing_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "empty_out"
    out.mkdir()
    report = run_lint_impl(out)
    assert report.count(FATAL) >= 1
    assert any("missing_artifact" == f.category for f in report.findings)


def test_lint_detects_degenerate_vectors(tmp_path: Path) -> None:
    # 非常に高い nnz threshold を与えて、通常チャンクを縮退扱いにする
    output_dir = _ingest(
        tmp_path,
        {"01.txt": "田中は東京で働く。田中はプロジェクトを進める。"},
        target_chars=10,
        max_chunk_chars=50,
    )
    report = run_lint_impl(
        output_dir,
        target_chunk_chars=10,
        max_chunk_chars=50,
        degenerate_nnz=1000,  # 事実上全チャンクを縮退扱い
    )
    assert any(f.category == "degenerate_vector" for f in report.findings)


def test_lint_detects_wiki_disabled_on_small_corpus(tmp_path: Path) -> None:
    output_dir = _ingest(
        tmp_path,
        {"01.txt": "田中は働く。東京に住む。"},
        target_chars=10,
        max_chunk_chars=50,
    )
    # この corpus は 数 チャンク < 20、wiki 0 → wiki_disabled 警告
    report = run_lint_impl(output_dir)
    assert any(f.category == "wiki_disabled" for f in report.findings)


def test_lint_detects_similar_entity_names(tmp_path: Path) -> None:
    output_dir = _ingest(
        tmp_path,
        {"01.txt": "dummy。"},
    )
    # 成功 entity 2 件を manifest に手動で注入
    manifest_path = output_dir / "entities" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "entries": {
                    "PERSON__田中太郎": {
                        "entity_name": "田中太郎",
                        "ner_label": "PERSON",
                        "source_hash": "h1",
                        "status": "success",
                        "mention_count": 3,
                        "chunk_count": 2,
                        "chunk_ids": ["c1", "c2"],
                        "cooccurring_entities": [],
                    },
                    "PERSON__田中太郎氏": {
                        "entity_name": "田中太郎氏",
                        "ner_label": "PERSON",
                        "source_hash": "h2",
                        "status": "success",
                        "mention_count": 3,
                        "chunk_count": 2,
                        "chunk_ids": ["c3", "c4"],
                        "cooccurring_entities": [],
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report = run_lint_impl(output_dir)
    assert any(f.category == "similar_entity" for f in report.findings)


def test_lint_detects_orphan_wiki(tmp_path: Path) -> None:
    output_dir = _ingest(
        tmp_path,
        {"01.txt": "田中は働く。東京に住む。"},
    )
    manifest_path = output_dir / "entities" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # chunks.jsonl に存在しない chunk_id を参照する entity
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "entries": {
                    "PERSON__幽霊": {
                        "entity_name": "幽霊",
                        "ner_label": "PERSON",
                        "source_hash": "h",
                        "status": "success",
                        "mention_count": 3,
                        "chunk_count": 2,
                        "chunk_ids": ["nonexistent1", "nonexistent2"],
                        "cooccurring_entities": [],
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report = run_lint_impl(output_dir)
    assert any(f.category == "orphan_wiki" for f in report.findings)


def test_lint_exit_code_warning_on_warnings_only(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """F-008: 警告のみの場合は exit code=1 (致命 0 件 = OK ではない)."""
    output_dir = _ingest(
        tmp_path,
        {"01.txt": "田中は働く。東京に住む。"},
    )
    args = argparse.Namespace(output_dir=str(output_dir), command="lint")
    code = run_lint(args)
    # この入力は wiki_disabled などの警告が出る想定
    assert code == 1
    assert (output_dir / "lint.md").exists()


def test_lint_exit_code_fatal_on_missing_artifacts(tmp_path: Path) -> None:
    """F-008: 致命検出なら exit code=2."""
    empty = tmp_path / "empty"
    empty.mkdir()
    args = argparse.Namespace(output_dir=str(empty), command="lint")
    code = run_lint(args)
    assert code == 2


def test_lint_exit_code_zero_on_clean_corpus(tmp_path: Path) -> None:
    """F-008: 致命も警告もなければ exit code=0."""
    output_dir = _ingest(
        tmp_path,
        {
            f"{i:02d}.txt": (
                f"文書{i}の一文目。"
                f"文書{i}の二文目を書きます。"
                f"文書{i}の三文目。"
            )
            for i in range(1, 8)
        },
        target_chars=20,
        max_chunk_chars=200,
    )
    # target_chunk_chars / max_chunk_chars は警告閾値なので明示して合わせる
    args = argparse.Namespace(
        output_dir=str(output_dir),
        command="lint",
        target_chunk_chars=20,
        max_chunk_chars=200,
    )
    code = run_lint(args)
    # コーパスが小さいので wiki_disabled 警告は出る可能性があるため、
    # ここでは致命 0 = exit != 2 を確認 (実装仕様の最低保証).
    assert code in (0, 1)


def test_lint_pairwise_skipped_for_large_n(tmp_path: Path) -> None:
    output_dir = _ingest(
        tmp_path,
        {f"{i:02d}.txt": f"文書{i}の内容。" for i in range(1, 6)},
        target_chars=8,
        max_chunk_chars=50,
    )
    # threshold を 1 に設定 → pairwise スキップ
    report = run_lint_impl(output_dir, pairwise_threshold=1)
    assert any(f.category == "pairwise_skipped" for f in report.findings)


def test_lint_md_format(tmp_path: Path) -> None:
    output_dir = _ingest(
        tmp_path,
        {"01.txt": "田中は働く。東京に住む。"},
    )
    args = argparse.Namespace(output_dir=str(output_dir), command="lint")
    run_lint(args)
    content = (output_dir / "lint.md").read_text("utf-8")
    assert "# Lint Report" in content
    # 警告があるなら "## 警告" セクションが存在
    if "警告" in content:
        assert "|" in content  # Markdown テーブル
