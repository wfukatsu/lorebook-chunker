"""U4: SkipReport + `skipped_files.jsonl` の動作テスト.

IngestRunner のファイル読み込みループが「空ファイル」「decode 失敗」「encoding 検出失敗」
などのスキップ理由を構造化した `SkipReport` として emit し、staging dir 内に
`skipped_files.jsonl` として (skip > 0 件のときのみ) 書き出すことを確認する.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from lorebook_chunker.ingest import IngestConfig, IngestRunner
from lorebook_chunker.schema import (
    SKIP_REASON_EMPTY_FILE,
    SKIP_REASON_ENCODING_DECODE_FAILED,
    SKIP_REASON_ENCODING_DETECTION_FAILED,
    SKIP_REASON_FILE_NOT_FOUND,
    SKIP_REASON_PERMISSION_DENIED,
    VALID_SKIP_REASONS,
    SkipReport,
)

# conftest の stub / fixture を使う
from tests.conftest import _ScriptedLLM, _StubAnalyzer


# ---- SkipReport dataclass ----------------------------------------------


def test_skip_report_to_jsonable_none_fields_become_null() -> None:
    """detail / encoding_attempted / size_bytes が None でも json.dumps で落ちない."""
    sr = SkipReport(path="/tmp/foo.txt", reason=SKIP_REASON_EMPTY_FILE)
    d = sr.to_jsonable()
    assert d == {
        "path": "/tmp/foo.txt",
        "reason": SKIP_REASON_EMPTY_FILE,
        "detail": None,
        "encoding_attempted": None,
        "size_bytes": None,
    }
    # JSON 化でも error なし, None は JSON null.
    serialized = json.dumps(d, ensure_ascii=False)
    assert '"detail": null' in serialized
    assert '"size_bytes": null' in serialized


def test_skip_report_to_jsonable_all_fields_populated() -> None:
    sr = SkipReport(
        path="/abs/path/bad.txt",
        reason=SKIP_REASON_ENCODING_DECODE_FAILED,
        detail="UnicodeDecodeError: ...",
        encoding_attempted="utf-8",
        size_bytes=123,
    )
    d = sr.to_jsonable()
    assert d["path"] == "/abs/path/bad.txt"
    assert d["reason"] == SKIP_REASON_ENCODING_DECODE_FAILED
    assert d["detail"] == "UnicodeDecodeError: ..."
    assert d["encoding_attempted"] == "utf-8"
    assert d["size_bytes"] == 123


@pytest.mark.parametrize(
    "reason",
    [
        SKIP_REASON_EMPTY_FILE,
        SKIP_REASON_ENCODING_DECODE_FAILED,
        SKIP_REASON_ENCODING_DETECTION_FAILED,
        SKIP_REASON_PERMISSION_DENIED,
        SKIP_REASON_FILE_NOT_FOUND,
    ],
)
def test_skip_report_reason_values_are_stable(reason: str) -> None:
    """Reason 値を契約として pin する. 値を変えたら下流 (U5 run_report.json) が壊れる."""
    assert reason in VALID_SKIP_REASONS


def test_valid_skip_reasons_contract_snapshot() -> None:
    """VALID_SKIP_REASONS の全体集合の snapshot."""
    assert VALID_SKIP_REASONS == frozenset(
        {
            "empty_file",
            "encoding_decode_failed",
            "encoding_detection_failed",
            "permission_denied",
            "file_not_found",
        }
    )


# ---- IngestRunner 統合: skip 0 件 / N 件のエンドツーエンド ----


def _runner_for(
    cfg: IngestConfig,
) -> IngestRunner:
    return IngestRunner(
        cfg,
        analyzer_factory=lambda path: _StubAnalyzer(),
        llm_factory=lambda backend, config: _ScriptedLLM([]),
    )


def test_skipped_files_jsonl_not_created_when_no_skips(
    tmp_path: Path,
    tmp_samples_dir: Callable,
    ingest_config_factory: Callable,
) -> None:
    """skip 0 件時は `skipped_files.jsonl` を作成しない (出力 dir を clean に保つ)."""
    input_dir = tmp_samples_dir(
        tmp_path / "in",
        {
            "ok.txt": "田中はスカラー商事で働いています。東京に住んでいます。",
        },
    )
    output_dir = tmp_path / "out"
    cfg = ingest_config_factory(
        tmp_path, input_dir=input_dir, output_dir=output_dir, encoding="utf-8"
    )
    result = _runner_for(cfg).run()
    assert result.exit_code == 0, result.errors
    assert len(result.skips) == 0
    assert result.skipped_files == 0  # property と一致
    assert not (output_dir / "skipped_files.jsonl").exists()


def test_skipped_files_jsonl_written_with_empty_and_bad_utf8(
    tmp_path: Path,
    ingest_config_factory: Callable,
) -> None:
    """空ファイル + bad-UTF-8 + OK 入力 → JSONL 2 行 / chunks.jsonl に OK のみ."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "empty.txt").write_text("", encoding="utf-8")
    (input_dir / "bad.txt").write_bytes(b"\xff\xfe\x00 invalid")
    (input_dir / "ok.txt").write_text(
        "田中はスカラー商事で働いています。東京に住んでいます。",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    # encoding="utf-8" を明示: bad.txt を決定論的に decode 失敗させる
    cfg = ingest_config_factory(
        tmp_path,
        input_dir=input_dir,
        output_dir=output_dir,
        encoding="utf-8",
    )
    result = _runner_for(cfg).run()

    assert result.exit_code == 0, result.errors
    assert len(result.skips) == 2
    assert result.skipped_files == 2  # 互換 property

    # JSONL ファイルが存在し、行数 == len(skips)
    jsonl_path = output_dir / "skipped_files.jsonl"
    assert jsonl_path.exists()
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(result.skips)

    # 各行は valid JSON で SkipReport shape を満たす
    records = [json.loads(line) for line in lines]
    for rec in records:
        assert set(rec.keys()) == {
            "path",
            "reason",
            "detail",
            "encoding_attempted",
            "size_bytes",
        }
        assert rec["reason"] in VALID_SKIP_REASONS

    reasons_in_jsonl = {rec["reason"] for rec in records}
    assert SKIP_REASON_EMPTY_FILE in reasons_in_jsonl
    # bad-UTF-8 は decode_failed か detection_failed のどちらかに分類される
    assert reasons_in_jsonl & {
        SKIP_REASON_ENCODING_DECODE_FAILED,
        SKIP_REASON_ENCODING_DETECTION_FAILED,
    }

    # warnings の back-compat substring は維持される
    assert any("empty file" in w for w in result.warnings)
    assert any("utf-8 decode failed" in w for w in result.warnings)

    # OK ファイルは chunks.jsonl に含まれる
    chunks_path = output_dir / "chunks.jsonl"
    assert chunks_path.exists()
    chunk_texts = [
        json.loads(line)["text"]
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any("田中" in t for t in chunk_texts)


def test_skipped_files_property_matches_skips_length(
    tmp_path: Path,
    ingest_config_factory: Callable,
) -> None:
    """`IngestResult.skipped_files` (互換 property) == `len(skips)`."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("", encoding="utf-8")
    (input_dir / "b.txt").write_text("", encoding="utf-8")
    (input_dir / "c.txt").write_text(
        "田中はスカラー商事で働いています。東京に住んでいます。",
        encoding="utf-8",
    )
    cfg = ingest_config_factory(
        tmp_path,
        input_dir=input_dir,
        output_dir=tmp_path / "out",
        encoding="utf-8",
    )
    result = _runner_for(cfg).run()
    assert result.exit_code == 0, result.errors
    assert result.skipped_files == len(result.skips) == 2


def test_recursive_skip_path_is_absolute(
    tmp_path: Path,
    ingest_config_factory: Callable,
) -> None:
    """recursive=True で配下から skip された場合、SkipReport.path は絶対パス文字列."""
    input_dir = tmp_path / "in"
    (input_dir / "nested" / "deep").mkdir(parents=True)
    (input_dir / "nested" / "deep" / "empty.txt").write_text("", encoding="utf-8")
    (input_dir / "ok.txt").write_text(
        "田中はスカラー商事で働いています。東京に住んでいます。",
        encoding="utf-8",
    )
    cfg = ingest_config_factory(
        tmp_path,
        input_dir=input_dir,
        output_dir=tmp_path / "out",
        recursive=True,
        encoding="utf-8",
    )
    result = _runner_for(cfg).run()
    assert result.exit_code == 0, result.errors
    assert len(result.skips) == 1
    sr = result.skips[0]
    assert sr.reason == SKIP_REASON_EMPTY_FILE
    # path は絶対 (Path.is_absolute で検証) かつ配下の nested/deep/empty.txt を指す
    assert Path(sr.path).is_absolute()
    assert "empty.txt" in sr.path
    assert "nested" in sr.path and "deep" in sr.path


def test_skip_report_encoding_decode_failure_context(
    tmp_path: Path,
    ingest_config_factory: Callable,
) -> None:
    """明示 encoding で decode 失敗 → reason=encoding_decode_failed + encoding_attempted 保存."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "bad.txt").write_bytes(b"\xff\xfe\x00 invalid byte sequence")
    (input_dir / "ok.txt").write_text(
        "田中はスカラー商事で働いています。東京に住んでいます。",
        encoding="utf-8",
    )
    cfg = ingest_config_factory(
        tmp_path,
        input_dir=input_dir,
        output_dir=tmp_path / "out",
        encoding="utf-8",
    )
    result = _runner_for(cfg).run()
    assert result.exit_code == 0, result.errors

    bad_skips = [s for s in result.skips if "bad.txt" in s.path]
    assert len(bad_skips) == 1
    sr = bad_skips[0]
    # encoding="utf-8" (明示) なので explicit_decode_failed ルートに入り、
    # encoding_decode_failed に translate される
    assert sr.reason == SKIP_REASON_ENCODING_DECODE_FAILED
    assert sr.encoding_attempted == "utf-8"
    assert sr.detail is not None  # exc message
    assert sr.size_bytes is not None and sr.size_bytes > 0


def test_skip_report_jsonl_line_count_matches_skips_length(
    tmp_path: Path,
    ingest_config_factory: Callable,
) -> None:
    """Integration: JSONL 行数 == len(result.skips). (U5 で run_report.json と cross-check される)."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("", encoding="utf-8")
    (input_dir / "b.txt").write_bytes(b"\xff\xfe\x00 invalid bytes here too")
    (input_dir / "c.txt").write_text("", encoding="utf-8")
    (input_dir / "d.txt").write_text(
        "田中はスカラー商事で働いています。東京に住んでいます。",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    cfg = ingest_config_factory(
        tmp_path,
        input_dir=input_dir,
        output_dir=output_dir,
        encoding="utf-8",
    )
    result = _runner_for(cfg).run()
    assert result.exit_code == 0, result.errors
    assert len(result.skips) == 3
    lines = (output_dir / "skipped_files.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(result.skips)


def test_all_jsonl_records_round_trip_through_json(
    tmp_path: Path,
    ingest_config_factory: Callable,
) -> None:
    """全 JSONL レコードが ensure_ascii=False で日本語 path を含め round-trip 可."""
    input_dir = tmp_path / "in" / "日本語ディレクトリ"
    input_dir.mkdir(parents=True)
    (input_dir / "空.txt").write_text("", encoding="utf-8")
    (input_dir / "ok.txt").write_text(
        "田中はスカラー商事で働いています。東京に住んでいます。",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    cfg = ingest_config_factory(
        tmp_path,
        input_dir=input_dir,
        output_dir=output_dir,
        encoding="utf-8",
    )
    result = _runner_for(cfg).run()
    assert result.exit_code == 0, result.errors
    assert len(result.skips) == 1

    raw = (output_dir / "skipped_files.jsonl").read_text(encoding="utf-8")
    # ensure_ascii=False のため生の日本語が含まれる
    assert "日本語" in raw or "空.txt" in raw
    rec = json.loads(raw.strip().splitlines()[0])
    assert rec["reason"] == SKIP_REASON_EMPTY_FILE
    assert "空.txt" in rec["path"]
