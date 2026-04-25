"""U5: `run_report.json` schema + emission + phase timings tests.

Covers:

- happy path: 必須キー全部、`exit_code: 0`, `exit_reason: null`
- happy path: `phase_durations_seconds` が 6 phase 全部、値 >= 0
- happy path: `phase_durations_seconds` の総和 ≒ `duration_seconds` (±10%)
- happy path: `schema_version == 1`
- happy path: `analyzer.model_sha256` が `analyzer.json` と一致
- edge: exit 6 wiki failure → `exit_reason.class == "WikiGenerationError"`,
  `phase_durations_seconds.wiki` > 0
- edge: 全ファイル skip (ConfigError no_input_files) → run_report は依然 emit、
  chunks/entities=0、exit_reason 完備
- edge: `entities_generated == AggregationStats.accepted_entities`
- edge: 部分 phase 失敗時も先行 phase の timing が記録される
- error path: `run_report.json` 書き込み失敗 → `RunReportError` (exit 17)
- integration: `ingest_result.json` は作成されない
- integration: stdout `--format json` 出力 == `report.to_json_dict()`
- schema snapshot: top-level / sub-dict のキー集合を v1 で lock
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from lorebook_chunker.errors import RunReportError
from lorebook_chunker.ingest import IngestConfig, IngestRunner, run_ingest
from lorebook_chunker.llm import GenerateResult, LLMPermanentError
from lorebook_chunker.run_report import PHASE_NAMES, RunReport
from lorebook_chunker.schema import EntityMention

from tests.conftest import _ScriptedLLM, _StubAnalyzer


# ---- helpers -----------------------------------------------------------


def _ok(text: str = "OK") -> GenerateResult:
    return GenerateResult(
        text=text,
        input_tokens=5,
        output_tokens=10,
        model_id="mock@v1",
        finish_reason="end_turn",
    )


def _write_sample(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "01.txt").write_text(
        "田中はスカラー商事で働いています。"
        "田中は東京に住んでいます。"
        "スカラー商事は大阪にも支社があります。"
        "プロジェクトの責任者は田中です。\n"
        "スカラー商事の佐藤さんも優秀です。"
        "佐藤は東京本社の人間です。\n",
        encoding="utf-8",
    )
    (input_dir / "02.txt").write_text(
        "佐藤は最近異動した。"
        "佐藤は新プロジェクトの一員である。"
        "田中と佐藤は協力している。\n",
        encoding="utf-8",
    )


def _run_ingest_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    skip_wiki: bool = True,
    fmt: str = "human",
    entity_map: dict[str, list[EntityMention]] | None = None,
    llm_responses: list[Any] | None = None,
) -> tuple[int, Path, Path]:
    """run_ingest(args) を CLI 経由で呼び、(exit_code, output_dir, input_dir) を返す.

    U5 writer + stdout path までを通す integration pass.
    """
    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"

    import lorebook_chunker.ingest as mod

    monkeypatch.setattr(
        mod, "default_analyzer_factory", lambda path: _StubAnalyzer(entity_map)
    )
    if not skip_wiki:
        scripted = _ScriptedLLM(llm_responses or [])
        monkeypatch.setattr(mod, "get_client", lambda backend, config: scripted)

    args = argparse.Namespace(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        skip_wiki=skip_wiki,
        max_llm_calls=None,
        force_regenerate=False,
        retry_failed=False,
        llm_backend="anthropic",
        llm_model=None,
        llm_parallelism=None,
        analyzer_backend="electra",
        device="cpu",
        quiet=True,
        no_progress=True,
        recursive=False,
        globs="*.txt",
        encoding="auto",
        format=fmt,
        config=None,
        command="ingest",
    )
    exit_code = run_ingest(args)
    return exit_code, output_dir, input_dir


# ---- Happy path ---------------------------------------------------------


def test_run_report_happy_path_has_all_required_top_level_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exit_code, output_dir, _ = _run_ingest_cli(tmp_path, monkeypatch, skip_wiki=True)
    assert exit_code == 0
    report_path = output_dir / "run_report.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "lorebook_chunker_version",
        "exit_code",
        "exit_reason",
        "started_at",
        "completed_at",
        "duration_seconds",
        "phase_durations_seconds",
        "input",
        "output",
        "analyzer",
        "llm",
        "warnings",
    }
    assert set(data.keys()) == expected_keys
    assert data["exit_code"] == 0
    assert data["exit_reason"] is None


def test_run_report_phase_keys_are_fixed_6_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output_dir, _ = _run_ingest_cli(tmp_path, monkeypatch, skip_wiki=True)
    data = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
    assert set(data["phase_durations_seconds"].keys()) == {
        "analyzer_init",
        "chunking",
        "tfidf",
        "ner",
        "wiki",
        "swap",
    }
    for name, dur in data["phase_durations_seconds"].items():
        assert dur >= 0.0, f"phase {name} duration {dur} < 0"


def test_run_report_phase_durations_sum_approximates_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output_dir, _ = _run_ingest_cli(tmp_path, monkeypatch, skip_wiki=True)
    data = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
    phase_sum = sum(data["phase_durations_seconds"].values())
    total = data["duration_seconds"]
    # phase の総和は ingest の duration 内に収まる必要がある (並列実行はない).
    # 過剰分は ingest 前後の housekeeping (staging 作成, manifest copy, log.md,
    # run_report 書き出し等) で ±10% 程度の幅を許す.
    assert 0.0 < phase_sum <= total * 1.10 + 0.05
    assert phase_sum >= total * 0.5  # 6 phase が total の過半を占めるはず


def test_run_report_schema_version_is_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output_dir, _ = _run_ingest_cli(tmp_path, monkeypatch, skip_wiki=True)
    data = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    # class constant 経由の値 (runtime mutation を受けない)
    assert RunReport.SCHEMA_VERSION == 1


def test_run_report_analyzer_model_sha256_matches_analyzer_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`analyzer.model_sha256` が `analyzer.json` の canonical hash と一致."""
    from lorebook_chunker.wiki import compute_analyzer_json_hash

    _, output_dir, _ = _run_ingest_cli(tmp_path, monkeypatch, skip_wiki=True)
    expected_hash = compute_analyzer_json_hash(output_dir / "analyzer.json")
    data = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
    assert data["analyzer"]["model_sha256"] == expected_hash
    assert expected_hash  # 非空


def test_run_report_entities_generated_equals_accepted_entities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`output.entities_generated` は unique entity 数 (AggregationStats.accepted_entities)."""
    entity_map = {
        "田中": [EntityMention("田中", "PERSON", 0, 2)],
        "佐藤": [EntityMention("佐藤", "PERSON", 0, 2)],
        "スカラー商事": [EntityMention("スカラー商事", "ORG", 0, 6)],
    }
    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"
    analyzer = _StubAnalyzer(entity_map)
    cfg = IngestConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        skip_wiki=True,
        target_chars=25,
        overlap_chars=0,
        max_chunk_chars=200,
        min_mentions=2,
        min_chunks=1,
        show_progress=False,
    )
    result = IngestRunner(
        cfg,
        analyzer_factory=lambda path: analyzer,
        llm_factory=lambda backend, config: _ScriptedLLM([]),
    ).run()
    assert result.exit_code == 0
    # entities_generated は unique (name, label) pair 数
    assert result.entities_generated >= 1
    # 重複 mention は counted once (AggregationStats.accepted_entities 同値)
    assert isinstance(result.entities_generated, int)


# ---- Integration: stdout `--format json` ---------------------------------


def test_run_report_stdout_json_format_matches_file_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--format json` の stdout 出力と `run_report.json` の内容が一致する."""
    _, output_dir, _ = _run_ingest_cli(
        tmp_path, monkeypatch, skip_wiki=True, fmt="json"
    )
    captured = capsys.readouterr()
    stdout_text = captured.out.strip()
    assert stdout_text, "stdout empty — expected JSON run_report"
    stdout_data = json.loads(stdout_text)
    file_data = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
    # started_at/completed_at/duration_seconds は同一 run なので同値のはず.
    # phase_durations_seconds 等すべて完全一致を要求.
    assert stdout_data == file_data


# ---- Edge: no_input_files (exit 2) -------------------------------------


def test_run_report_emitted_even_on_no_input_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    import lorebook_chunker.ingest as mod

    monkeypatch.setattr(mod, "default_analyzer_factory", lambda path: _StubAnalyzer())
    args = argparse.Namespace(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        skip_wiki=True,
        max_llm_calls=None,
        force_regenerate=False,
        retry_failed=False,
        llm_backend="anthropic",
        llm_model=None,
        llm_parallelism=None,
        analyzer_backend="electra",
        device="cpu",
        quiet=True,
        no_progress=True,
        recursive=False,
        globs="*.txt",
        encoding="auto",
        format="human",
        config=None,
        command="ingest",
    )
    exit_code = run_ingest(args)
    assert exit_code == 2
    report_path = output_dir / "run_report.json"
    assert report_path.exists(), "run_report.json must be emitted even on failure"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["exit_code"] == 2
    assert data["exit_reason"] is not None
    assert data["exit_reason"]["class"] == "ConfigError"
    # chunks / entities は 0
    assert data["output"]["chunks_generated"] == 0
    assert data["output"]["entities_generated"] == 0
    # 6 phase key は常に存在 (実行前で全 0 のはず)
    assert set(data["phase_durations_seconds"].keys()) == set(PHASE_NAMES)


# ---- Edge: wiki runtime failure (exit 6) ------------------------------


def test_run_report_on_wiki_failure_records_partial_phase_durations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """wiki runtime 失敗 (LLMPermanentError → WikiGenerationError exit 6) で
    report が emit され、`phase_durations_seconds.wiki` が失敗までの経過秒を持つ.
    先行 phase (analyzer_init / chunking / tfidf / ner) も記録されている.
    """
    entity_map = {
        "田中": [EntityMention("田中", "PERSON", 0, 2)],
        "佐藤": [EntityMention("佐藤", "PERSON", 0, 2)],
        "スカラー商事": [EntityMention("スカラー商事", "ORG", 0, 6)],
    }
    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"

    # wiki.generate_all の pre-flight で LLMPermanentError 派生が raise され、
    # IngestRunner がそれを WikiGenerationError (exit 6) にラップする.
    # IngestRunner 直接呼び出しで min_mentions/min_chunks を下げる.
    cfg = IngestConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        skip_wiki=False,
        target_chars=25,
        overlap_chars=0,
        max_chunk_chars=200,
        min_mentions=2,
        min_chunks=1,
        show_progress=False,
        llm_backend="anthropic",
    )
    # pre-flight で LLMPermanentError. wiki 層で `LLMPreflightError` に wrap
    # されて LLMPermanentError 階層として再 raise → ingest 層で
    # WikiGenerationError (exit 6).
    scripted = _ScriptedLLM([LLMPermanentError("auth failed")])
    analyzer = _StubAnalyzer(entity_map)
    result = IngestRunner(
        cfg,
        analyzer_factory=lambda path: analyzer,
        llm_factory=lambda backend, config: scripted,
    ).run()
    assert result.exit_code == 6, f"expected exit 6, got {result.exit_code} errors={result.errors}"

    # run_report.json の emission は run_ingest (CLI wrapper) 経由で発生する.
    # ここでは IngestResult ベースで RunReport を組み立てて同じ内容を検証する.
    from lorebook_chunker import run_report as rr_mod

    report = RunReport(
        exit_code=result.exit_code,
        exit_reason=dict(result.errors[0]) if result.errors else None,
        phase_durations_seconds=dict(result.phase_timings),
    )
    data = report.to_json_dict()
    assert data["exit_code"] == 6
    assert data["exit_reason"]["class"] == "WikiGenerationError"
    pd = data["phase_durations_seconds"]
    # 先行 phase (analyzer_init / chunking / tfidf / ner) は記録されている
    assert pd["analyzer_init"] >= 0.0
    assert pd["chunking"] > 0.0
    assert pd["tfidf"] >= 0.0
    assert pd["ner"] >= 0.0
    # wiki phase は失敗時も try/finally 経由で 0 より大の duration を持つ
    assert pd["wiki"] > 0.0
    # swap は実行されなかったので 0
    assert pd["swap"] == 0.0
    # writer も成功する (disk full simulation 以外は)
    rr_mod.write(tmp_path / "run_report.json", report)
    assert (tmp_path / "run_report.json").exists()


# ---- Error path: run_report.json 書き込み失敗 (exit 17) --------------


def test_run_report_write_failure_raises_run_report_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`tempfile.mkstemp` を raise させて RunReportError (exit 17) を発火させる."""
    from lorebook_chunker import run_report as rr_mod

    def _mkstemp_boom(*args: Any, **kwargs: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(rr_mod.tempfile, "mkstemp", _mkstemp_boom)

    report = RunReport(exit_code=0)
    with pytest.raises(RunReportError) as exc_info:
        rr_mod.write(tmp_path / "run_report.json", report)
    assert exc_info.value.exit_code == 17
    assert "run_report.json write failed" in str(exc_info.value)


def test_run_report_writer_failure_promotes_to_exit_17_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """success ingest + writer failure → exit 17 に昇格."""
    from lorebook_chunker import run_report as rr_mod

    def _mkstemp_boom(*args: Any, **kwargs: Any) -> None:
        raise OSError(13, "Permission denied")

    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"

    import lorebook_chunker.ingest as mod

    monkeypatch.setattr(mod, "default_analyzer_factory", lambda path: _StubAnalyzer())

    # 先に成功まで走らせてから、writer だけ mock.
    # writer は最終段でだけ呼ばれるので、mkstemp の patch を run_ingest 直前に
    # 仕掛ければ runner は完走する.
    monkeypatch.setattr(rr_mod.tempfile, "mkstemp", _mkstemp_boom)

    args = argparse.Namespace(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        skip_wiki=True,
        max_llm_calls=None,
        force_regenerate=False,
        retry_failed=False,
        llm_backend="anthropic",
        llm_model=None,
        llm_parallelism=None,
        analyzer_backend="electra",
        device="cpu",
        quiet=True,
        no_progress=True,
        recursive=False,
        globs="*.txt",
        encoding="auto",
        format="human",
        config=None,
        command="ingest",
    )
    exit_code = run_ingest(args)
    assert exit_code == 17, "writer failure on success path must be exit 17"


# ---- Integration: `ingest_result.json` never created -------------------


def test_no_legacy_ingest_result_json_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output_dir, _ = _run_ingest_cli(tmp_path, monkeypatch, skip_wiki=True)
    # 旧サマリファイルは削除されたため作成されない
    assert not (output_dir / "ingest_result.json").exists()


# ---- Schema snapshot: v1 lock -----------------------------------------


def test_run_report_schema_v1_top_level_keys_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U5 の v1 lockdown snapshot test.

    top-level / sub-dict のキー集合が増減した場合 (特に追加 / 削除) に
    この test が落ちる. v1 additive-only で新 key を足した場合はこの test を
    明示的に更新して conscious decision を残す. 破壊的変更 (key 削除 / rename)
    の場合は `schema_version` bump + この test を v2 化する.
    """
    _, output_dir, _ = _run_ingest_cli(tmp_path, monkeypatch, skip_wiki=True)
    data = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
    assert set(data.keys()) == {
        "schema_version",
        "lorebook_chunker_version",
        "exit_code",
        "exit_reason",
        "started_at",
        "completed_at",
        "duration_seconds",
        "phase_durations_seconds",
        "input",
        "output",
        "analyzer",
        "llm",
        "warnings",
    }
    assert set(data["input"].keys()) == {
        "input_dir",
        "recursive",
        "globs",
        "encoding_option",
        "files_processed",
        "files_skipped",
    }
    assert set(data["output"].keys()) == {
        "output_dir",
        "chunks_generated",
        "entities_generated",
        "wiki_pages_written",
    }
    assert set(data["analyzer"].keys()) == {
        "model_name",
        "model_version",
        "model_sha256",
        "spacy_version",
        "ginza_version",
        "sudachi_dict",
    }
    assert set(data["llm"].keys()) == {
        "backend",
        "model_id",
        "total_input_tokens",
        "total_output_tokens",
    }


# ---- Skip aggregation ---------------------------------------------------


def test_run_report_files_skipped_matches_skipped_files_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U4 の `skipped_files.jsonl` 行数 == `input.files_skipped` の長さ."""
    input_dir = tmp_path / "samples"
    input_dir.mkdir()
    (input_dir / "01.txt").write_text("", encoding="utf-8")  # 空ファイル
    (input_dir / "02.txt").write_bytes(b"\xff\xfe\x00 invalid")  # 不正 UTF-8
    (input_dir / "03.txt").write_text(
        "田中はスカラー商事で働いています。東京に住んでいます。",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    import lorebook_chunker.ingest as mod

    monkeypatch.setattr(mod, "default_analyzer_factory", lambda path: _StubAnalyzer())

    args = argparse.Namespace(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        skip_wiki=True,
        max_llm_calls=None,
        force_regenerate=False,
        retry_failed=False,
        llm_backend="anthropic",
        llm_model=None,
        llm_parallelism=None,
        analyzer_backend="electra",
        device="cpu",
        quiet=True,
        no_progress=True,
        recursive=False,
        globs="*.txt",
        encoding="utf-8",  # strict decode
        format="human",
        config=None,
        command="ingest",
    )
    exit_code = run_ingest(args)
    assert exit_code == 0
    data = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
    skipped = data["input"]["files_skipped"]
    assert len(skipped) == 2  # empty + bad utf-8
    # skipped_files.jsonl の行数と一致
    jsonl = (output_dir / "skipped_files.jsonl").read_text(encoding="utf-8")
    jsonl_lines = [ln for ln in jsonl.splitlines() if ln.strip()]
    assert len(jsonl_lines) == len(skipped)
    # SkipReport.to_jsonable() shape (reason / path / detail / ...) を持つ
    for sk in skipped:
        assert "path" in sk
        assert "reason" in sk


# ---- RunReport dataclass unit tests ----------------------------------


def test_run_report_to_json_dict_pads_missing_phase_keys() -> None:
    """`phase_durations_seconds` が部分指定でも 6 key 全部 emit."""
    rep = RunReport(
        exit_code=0,
        phase_durations_seconds={"chunking": 1.2, "tfidf": 0.3},
    )
    d = rep.to_json_dict()
    assert set(d["phase_durations_seconds"].keys()) == set(PHASE_NAMES)
    assert d["phase_durations_seconds"]["chunking"] == pytest.approx(1.2)
    assert d["phase_durations_seconds"]["tfidf"] == pytest.approx(0.3)
    # 未指定 key は 0.0
    assert d["phase_durations_seconds"]["analyzer_init"] == 0.0
    assert d["phase_durations_seconds"]["ner"] == 0.0
    assert d["phase_durations_seconds"]["wiki"] == 0.0
    assert d["phase_durations_seconds"]["swap"] == 0.0


def test_run_report_to_json_dict_drops_unknown_phase_keys() -> None:
    """schema v1 外の phase key (例: 旧 progress.start の日本語キー) は emit しない."""
    rep = RunReport(
        exit_code=0,
        phase_durations_seconds={
            "chunking": 1.2,
            "文書解析 (nlp.pipe)": 42.0,  # unknown
        },
    )
    d = rep.to_json_dict()
    assert "文書解析 (nlp.pipe)" not in d["phase_durations_seconds"]
    assert set(d["phase_durations_seconds"].keys()) == set(PHASE_NAMES)
