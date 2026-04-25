"""U1: Error taxonomy + exit-code table + CLI boundary tests.

このテストは 2 layered になっている:

1. Characterization tests — 既存の exit-code 挙動 (2/3/4/5/6/10) を lock する.
   これらのテストは refactor 前後の両方で pass する必要がある. Refactor が
   exit-code を silently break した場合に最初に気づけるようにする.

2. New taxonomy tests — `errors.LorebookError` 階層を検証する:
   - `to_jsonable()` round-trip
   - non-serializable context の str() degrade
   - `describe_exit_codes()` が EXIT_CODE_MAP 全エントリを含む
   - CLI boundary (`run_ingest`) の dispatch 順 (`LorebookError` branch が
     generic `Exception` branch より先)
   - `INGEST_EXIT_CODES` 文字列リテラルが src/ 配下で errors.py のみに存在
   - `except RuntimeError` で retrofit 後の schema.py 例外を捕捉する caller が
     存在しないこと (direct-inheritance decision の安全性根拠)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import pytest

from lorebook_chunker.chunker import simple_japanese_splitter
from lorebook_chunker.ingest import IngestConfig, IngestRunner, run_ingest
from lorebook_chunker.llm import (
    GenerateResult,
    LLMPermanentError,
    LLMPreflightError,
)
from lorebook_chunker.schema import AnalyzerConfig, EntityMention


# ---- Test-local stubs (duplicated from tests/test_ingest.py; U2 will
#      consolidate into conftest.py). --------------------------------


class _StubAnalyzer:
    def __init__(self, entity_map: dict[str, list[EntityMention]] | None = None) -> None:
        self._entities = entity_map or {}

    def iter_sentences(self, text: str) -> Iterable[str]:
        return simple_japanese_splitter(text)

    def iter_entities(self, text: str) -> Iterable[EntityMention]:
        for key, ents in self._entities.items():
            if key in text:
                yield from ents

    def tokenize_for_tfidf(self, text: str) -> list[str]:
        tokens: list[str] = []
        for chunk in text.replace("\n", "。").split("。"):
            for word in chunk.split():
                w = word.strip("、.,")
                if len(w) >= 2:
                    tokens.append(w)
        for word in ["田中", "佐藤", "スカラー商事", "東京", "大阪", "プロジェクト"]:
            count = text.count(word)
            for _ in range(count):
                tokens.append(word)
        return tokens

    def save(self, path: Any) -> None:
        import json as _json

        config = AnalyzerConfig(
            strict_match={
                "model_name": "stub",
                "split_mode": "STUB",
                "pos_allowlist": ["NOUN"],
                "stopwords": [],
                "lemma_rules": "stub",
                "normalization": {
                    "nfkc": True,
                    "lf_only": True,
                    "strip_trailing": True,
                    "collapse_spaces": True,
                },
                "sudachidict_binary_sha256": "stub",
                "model_checksum": "stub",
            },
            compat_match={"ginza": "stub-0.0", "spacy": "stub-0.0", "sudachipy": "stub-0.0"},
            tfidf={"min_df": 1, "max_df": 0.95},
        )
        Path(path).write_text(_json.dumps(config.to_dict(), ensure_ascii=False), encoding="utf-8")

    def build_config(self) -> AnalyzerConfig:
        return AnalyzerConfig(
            strict_match={"model_name": "stub"},
            compat_match={"ginza": "stub-0.0"},
            tfidf={},
        )


class _ScriptedLLM:
    model_id = "mock@v1"

    def __init__(self, responses: list[Any] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls = 0

    def generate(self, prompt: str, max_tokens: int) -> GenerateResult:
        self.calls += 1
        r = self._responses.pop(0) if self._responses else GenerateResult(
            text="OK",
            input_tokens=5,
            output_tokens=5,
            model_id="mock@v1",
            finish_reason="end_turn",
        )
        if isinstance(r, Exception):
            raise r
        return r


def _write_sample(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "01.txt").write_text(
        "田中はスカラー商事で働いています。"
        "田中は東京に住んでいます。"
        "スカラー商事は大阪にも支社があります。"
        "プロジェクトの責任者は田中です。\n"
        "スカラー商事の佐藤さんも優秀です。"
        "佐藤は東京本社の人間です。"
        "プロジェクトは半年続きます。\n",
        encoding="utf-8",
    )


# ====================================================================
# Section 1: Characterization tests (exit-code lockdown)
# ====================================================================
#
# これらは「refactor 前の exit-code 挙動が refactor 後も変わらない」
# ことを確認する regression gate. exit-code 数値そのものを checked-in.


def test_characterization_no_input_files_exits_2(tmp_path: Path) -> None:
    """R6: 入力ファイルが無い → exit 2 (既存契約)."""
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    cfg = IngestConfig(input_dir=input_dir, output_dir=output_dir, skip_wiki=True)
    runner = IngestRunner(
        cfg,
        analyzer_factory=lambda path: _StubAnalyzer(),
        llm_factory=lambda backend, config: _ScriptedLLM([]),
    )
    result = runner.run()
    assert result.exit_code == 2
    # 既存テスト test_ingest_empty_input_fails も "no .txt" substring に依存.
    assert any("no .txt" in _error_message(e) for e in result.errors)


def test_characterization_preflight_llm_fail_exits_3(tmp_path: Path) -> None:
    """R6: preflight LLM 初期化失敗 → exit 3 (既存契約)."""
    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"
    cfg = IngestConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        skip_wiki=False,
        llm_backend="anthropic",
    )

    def _bad_llm_factory(backend: str, config: dict[str, Any] | None) -> Any:
        raise LLMPermanentError("authentication failed (mock)")

    runner = IngestRunner(
        cfg,
        analyzer_factory=lambda path: _StubAnalyzer(),
        llm_factory=_bad_llm_factory,
    )
    result = runner.run()
    assert result.exit_code == 3


def test_characterization_analyzer_init_fail_exits_4(tmp_path: Path) -> None:
    """R6: analyzer init 失敗 → exit 4 (既存契約)."""
    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"
    cfg = IngestConfig(input_dir=input_dir, output_dir=output_dir, skip_wiki=True)

    def _bad_analyzer_factory(path: Any) -> Any:
        raise RuntimeError("spaCy model load failed (mock)")

    runner = IngestRunner(
        cfg,
        analyzer_factory=_bad_analyzer_factory,
        llm_factory=lambda backend, config: _ScriptedLLM([]),
    )
    result = runner.run()
    assert result.exit_code == 4


def test_characterization_zero_chunks_exits_5(tmp_path: Path) -> None:
    """R6: chunking produced 0 chunks → exit 5 (既存契約)."""
    input_dir = tmp_path / "samples"
    input_dir.mkdir()
    # 内容は入るが空白のみで normalize 後も chunk が 0 になるように仕向ける
    # — 代わりに stub analyzer が空 sentences を返すようにする.
    (input_dir / "01.txt").write_text("   ", encoding="utf-8")  # whitespace only
    (input_dir / "02.txt").write_text("   ", encoding="utf-8")
    output_dir = tmp_path / "out"

    class _EmptyAnalyzer(_StubAnalyzer):
        def iter_sentences(self, text: str):  # type: ignore[override]
            return []  # sentence splitter が何も返さない → 0 chunks

    cfg = IngestConfig(input_dir=input_dir, output_dir=output_dir, skip_wiki=True)
    runner = IngestRunner(
        cfg,
        analyzer_factory=lambda path: _EmptyAnalyzer(),
        llm_factory=lambda backend, config: _ScriptedLLM([]),
    )
    result = runner.run()
    # ファイル内容が whitespace only の場合 empty file として skip される可能性もあり、
    # その場合 0 files processed → "no chunks" ではなく earlier path で止まる.
    # ingest.py 実装: empty file は skipped_files に加算されるが input_files は
    # nonempty なので chunking phase まで到達する.
    assert result.exit_code == 5


def test_characterization_wiki_runtime_fail_exits_6(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R6: wiki.generate_all 実行時失敗 → exit 6 (既存契約).

    `WikiGenerator.generate_all` を monkeypatch で `LLMPermanentError` を
    raise させる (systemic abort の path を simulate). ingest.py:521-526 の
    except 句で exit 6 になることを確認.
    """
    import lorebook_chunker.ingest as ingest_mod

    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"

    entities = {
        "田中": [
            EntityMention(name="田中", ner_label="PERSON", char_start=0, char_end=2)
        ],
        "佐藤": [
            EntityMention(name="佐藤", ner_label="PERSON", char_start=0, char_end=2)
        ],
        "スカラー商事": [
            EntityMention(name="スカラー商事", ner_label="ORG", char_start=0, char_end=6)
        ],
    }

    def _always_raise(self, aggregates):  # type: ignore[no-untyped-def]
        raise LLMPermanentError("mock systemic wiki failure")

    monkeypatch.setattr(
        "lorebook_chunker.wiki.WikiGenerator.generate_all", _always_raise
    )

    cfg = IngestConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        skip_wiki=False,
        max_llm_calls=20,
        min_mentions=1,
        min_chunks=1,
        target_chars=30,
        max_chunk_chars=200,
    )
    runner = IngestRunner(
        cfg,
        analyzer_factory=lambda path: _StubAnalyzer(entity_map=entities),
        llm_factory=lambda backend, config: _ScriptedLLM([]),
    )
    result = runner.run()
    assert result.exit_code == 6


def test_characterization_unclassified_exception_exits_10(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R6: 未分類例外が CLI boundary に到達 → exit 10 fallback."""
    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"

    # chunker 中で未分類 ValueError を投げる stub.
    # iter_sentences を通るが tokenize_for_tfidf で unexpected error.
    class _ExplodingAnalyzer(_StubAnalyzer):
        def tokenize_for_tfidf(self, text: str) -> list[str]:  # type: ignore[override]
            raise ValueError("mock unclassified failure")

    cfg = IngestConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        skip_wiki=True,
        target_chars=30,
        max_chunk_chars=200,
    )
    runner = IngestRunner(
        cfg,
        analyzer_factory=lambda path: _ExplodingAnalyzer(),
        llm_factory=lambda backend, config: _ScriptedLLM([]),
    )
    result = runner.run()
    assert result.exit_code == 10


# ====================================================================
# Section 2: New taxonomy tests — LorebookError hierarchy
# ====================================================================


def test_errors_module_defines_expected_hierarchy() -> None:
    """errors.py が仕様通りのクラス階層を export している."""
    from lorebook_chunker import errors

    # base
    assert issubclass(errors.LorebookError, Exception)
    assert errors.LorebookError.exit_code == 10

    # per-category exit_code
    assert errors.ConfigError.exit_code == 2
    assert errors.LLMBackendUnavailableError.exit_code == 3
    assert errors.AnalyzerInitError.exit_code == 4
    assert errors.WikiGenerationError.exit_code == 6
    assert errors.InputError.exit_code == 11
    assert errors.EncodingError.exit_code == 12
    assert errors.ChunkingError.exit_code == 13
    assert errors.TfidfError.exit_code == 14
    assert errors.EntityAggregationError.exit_code == 15
    assert errors.AtomicSwapError.exit_code == 16
    assert errors.RunReportError.exit_code == 17
    # ZeroChunksError は exit 5 を保持 (既存契約)
    assert errors.ZeroChunksError.exit_code == 5


@pytest.mark.parametrize(
    "error_cls, kwargs",
    [
        ("ConfigError", {"reason": "no_input_files", "input_dir": "/tmp/x"}),
        ("LLMBackendUnavailableError", {"backend": "anthropic", "reason": "auth"}),
        ("AnalyzerInitError", {"reason": "model_load_failed"}),
        ("ZeroChunksError", {"reason": "zero_chunks"}),
        ("WikiGenerationError", {"reason": "systemic_abort", "failed": 3, "attempted": 5}),
        ("InputError", {"reason": "permission_denied", "path": "/tmp/x.txt"}),
        ("EncodingError", {"path": "/tmp/x.txt", "encoding_attempted": "utf-8"}),
        ("ChunkingError", {"reason": "empty_result"}),
        ("TfidfError", {"reason": "dimension_mismatch"}),
        ("EntityAggregationError", {"reason": "threshold_config_invalid"}),
        ("AtomicSwapError", {"source": "/tmp/a", "target": "/tmp/b", "errno": 18}),
        ("RunReportError", {"reason": "write_failed", "path": "/tmp/run_report.json"}),
    ],
)
def test_errors_to_jsonable_roundtrip(error_cls: str, kwargs: dict[str, Any]) -> None:
    """各 LorebookError サブクラスの to_jsonable() が round-trip 可能."""
    import json as _json

    from lorebook_chunker import errors

    cls = getattr(errors, error_cls)
    err = cls("example message", **kwargs)
    payload = err.to_jsonable()
    assert payload["class"] == error_cls
    assert payload["message"] == "example message"
    assert payload["context"] == {k: _stringify_if_needed(v) for k, v in kwargs.items()}
    # JSON serialize-able (ensure_ascii=False)
    _json.dumps(payload, ensure_ascii=False)


def _stringify_if_needed(v: Any) -> Any:
    # test helper: simple passthrough for str/int/float
    if isinstance(v, (str, int, float)):
        return v
    return str(v)


def test_errors_to_jsonable_degrades_nonserializable_context(tmp_path: Path) -> None:
    """context に Path や bytes など non-primitive が入ると str() で degrade."""
    from lorebook_chunker import errors

    err = errors.EncodingError(
        "decode failed",
        path=tmp_path / "x.txt",  # Path
        sample=b"\xff\xfe",  # bytes
    )
    payload = err.to_jsonable()
    assert payload["class"] == "EncodingError"
    # path は str() で degrade
    assert isinstance(payload["context"]["path"], str)
    assert str(tmp_path) in payload["context"]["path"]
    assert isinstance(payload["context"]["sample"], str)
    # JSON 化が raise しない
    import json as _json

    _json.dumps(payload, ensure_ascii=False)


def test_errors_describe_exit_codes_contains_all_entries() -> None:
    """describe_exit_codes() が EXIT_CODE_MAP 全エントリを含む."""
    from lorebook_chunker import errors

    text = errors.describe_exit_codes()
    for cls, code in errors.EXIT_CODE_MAP.items():
        assert cls.__name__ in text, f"missing class {cls.__name__} in describe_exit_codes()"
        assert str(code) in text, f"missing code {code} in describe_exit_codes()"


def test_errors_exit_code_map_covers_expected_codes() -> None:
    """EXIT_CODE_MAP が 2/3/4/5/6/11-17 全てを含む (後方互換 + 新規)."""
    from lorebook_chunker import errors

    codes = set(errors.EXIT_CODE_MAP.values())
    # 既存契約
    assert {2, 3, 4, 5, 6}.issubset(codes)
    # 新規 11-17
    assert {11, 12, 13, 14, 15, 16, 17}.issubset(codes)


def test_atomic_swap_error_caught_by_lorebook_error_branch(tmp_path: Path) -> None:
    """AtomicSwapError (exit 16) が ingest の CLI boundary で LorebookError
    dispatch (exit_code=16) に落ちることを確認.

    dispatch 順を間違えて `except Exception` が先に捕まえていれば exit 10 に
    なってしまう — この regression gate はそれを防ぐ.
    """
    from lorebook_chunker import errors

    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"

    # _atomic_swap を monkeypatch して AtomicSwapError を raise させる
    import lorebook_chunker.ingest as ingest_mod

    def _boom(target: Path, staging: Path, *, verify: bool = False) -> None:
        raise errors.AtomicSwapError(
            "mock swap failure",
            reason="EXDEV",
            source=str(staging),
            target=str(target),
            errno=18,
        )

    original = ingest_mod._atomic_swap
    ingest_mod._atomic_swap = _boom  # type: ignore[assignment]
    try:
        cfg = IngestConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            skip_wiki=True,
            target_chars=30,
            max_chunk_chars=200,
        )
        runner = IngestRunner(
            cfg,
            analyzer_factory=lambda path: _StubAnalyzer(),
            llm_factory=lambda backend, config: _ScriptedLLM([]),
        )
        result = runner.run()
    finally:
        ingest_mod._atomic_swap = original  # type: ignore[assignment]

    assert result.exit_code == 16
    # typed error: errors[0] は dict 形式 (LorebookError dispatch)
    assert result.errors, "no errors recorded"
    first = result.errors[0]
    # LorebookError path → dict with class/message/context
    assert isinstance(first, dict)
    assert first["class"] == "AtomicSwapError"
    assert "context" in first


def test_no_duplicate_ingest_exit_codes_literal_in_src() -> None:
    """INGEST_EXIT_CODES 文字列リテラルは errors.py (または describe_exit_codes
    の内部) のみに残っている — cli.py や ingest.py に重複が残っていない."""
    src_root = Path(__file__).resolve().parent.parent / "src"
    cli_py = src_root / "lorebook_chunker" / "cli.py"
    ingest_py = src_root / "lorebook_chunker" / "ingest.py"

    assert "INGEST_EXIT_CODES" not in cli_py.read_text(encoding="utf-8"), (
        "cli.py should not contain INGEST_EXIT_CODES literal (moved to errors.py)"
    )
    assert "INGEST_EXIT_CODES" not in ingest_py.read_text(encoding="utf-8"), (
        "ingest.py should not contain INGEST_EXIT_CODES literal (moved to errors.py)"
    )


def test_no_caller_catches_schema_errors_via_runtime_error() -> None:
    """schema.py の retrofit 対象 3 クラスを catch する `except RuntimeError`
    が src/ および tests/ 配下に残っていない (direct-inheritance 方針の
    安全性根拠).

    AST 解析で `except RuntimeError:` ハンドラだけを検出する (コメントや
    docstring の文字列一致は無視する).
    """
    import ast

    search_roots = [
        Path(__file__).resolve().parent.parent / "src",
        Path(__file__).resolve().parent,
    ]
    hits: list[str] = []
    for root in search_roots:
        for p in root.rglob("*.py"):
            try:
                tree = ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                exc_type = node.type
                # except RuntimeError:
                if isinstance(exc_type, ast.Name) and exc_type.id == "RuntimeError":
                    hits.append(f"{p}:{node.lineno}")
                # except (RuntimeError, ...):
                elif isinstance(exc_type, ast.Tuple):
                    for elt in exc_type.elts:
                        if isinstance(elt, ast.Name) and elt.id == "RuntimeError":
                            hits.append(f"{p}:{node.lineno}")
                            break
    assert not hits, f"unexpected `except RuntimeError` callers: {hits}"


def test_schema_errors_inherit_from_lorebook_error() -> None:
    """schema.py の 3 例外が LorebookError 派生に retrofit されている."""
    from lorebook_chunker import errors
    from lorebook_chunker.schema import (
        AnalyzerNEUnavailableError,
        AnalyzerVersionMismatchError,
        ChunkFileCorruptError,
    )

    assert issubclass(AnalyzerVersionMismatchError, errors.LorebookError)
    assert issubclass(AnalyzerNEUnavailableError, errors.LorebookError)
    assert issubclass(ChunkFileCorruptError, errors.LorebookError)


def test_llm_error_hierarchy_compatible_with_lorebook_error() -> None:
    """LLMError 階層が LorebookError 派生になっている (既存 isinstance checks
    を壊さないように LLMError 自体を LorebookError に inherit させる)."""
    from lorebook_chunker import errors
    from lorebook_chunker.llm import (
        LLMError,
        LLMPermanentError,
        LLMPreflightError,
        LLMRetryableError,
    )

    assert issubclass(LLMError, errors.LorebookError)
    assert issubclass(LLMPermanentError, LLMError)
    assert issubclass(LLMRetryableError, LLMError)
    assert issubclass(LLMPreflightError, LLMPermanentError)


def test_cli_ingest_help_epilog_mentions_new_exit_codes() -> None:
    """lorebook-chunker ingest --help の epilog で新 exit codes (11-17) が
    表示される — errors.describe_exit_codes() から生成されている証左."""
    parser_out = _capture_help(["ingest", "--help"])
    # 既存 2/3/4/5/6/10 は維持
    for code in ("2", "3", "4", "5", "6", "10"):
        assert re.search(rf"\b{code}\b", parser_out), f"missing exit code {code}"
    # 新規 11-17 も登場
    for code in ("11", "12", "13", "14", "15", "16", "17"):
        assert re.search(rf"\b{code}\b", parser_out), f"missing exit code {code}"


def _capture_help(argv: list[str]) -> str:
    from lorebook_chunker.cli import build_parser

    parser = build_parser()
    try:
        parser.parse_args(argv)
    except SystemExit:
        pass
    # re-run with capsys-like approach: parser.format_help on the subparser
    # — simpler path: use argparse's print via a known subparser lookup.
    # For a robust capture, we call the subparser's format_help() directly.
    subparsers_action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    sub = subparsers_action.choices[argv[0]]
    return sub.format_help()


def _error_message(err: Any) -> str:
    """IngestResult.errors は refactor 前 `list[str]`、refactor 後 `list[dict]`.
    どちらでも substring マッチできるように文字列化する."""
    if isinstance(err, dict):
        parts = [err.get("class", ""), err.get("message", "")]
        ctx = err.get("context") or {}
        parts.extend(f"{k}={v}" for k, v in ctx.items())
        return " ".join(str(p) for p in parts)
    return str(err)
