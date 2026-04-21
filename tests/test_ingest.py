"""IngestRunner のテスト. stub analyzer + mock LLM でフルパイプラインを回す."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pytest

from lorebook_chunker.chunker import simple_japanese_splitter
from lorebook_chunker.ingest import IngestConfig, IngestRunner, run_ingest
from lorebook_chunker.llm import GenerateResult, LLMPermanentError
from lorebook_chunker.schema import AnalyzerConfig, EntityMention


class _StubAnalyzer:
    """Ginza 非依存の stub. simple_japanese_splitter を使い、固定の NER を返す."""

    def __init__(self, entity_map: dict[str, list[EntityMention]] | None = None) -> None:
        self._entities = entity_map or {}

    def iter_sentences(self, text: str) -> Iterable[str]:
        return simple_japanese_splitter(text)

    def iter_entities(self, text: str) -> Iterable[EntityMention]:
        # 入力 text の最初のキー (substring) がマッチすればそのエンティティを返す
        for key, ents in self._entities.items():
            if key in text:
                yield from ents

    def tokenize_for_tfidf(self, text: str) -> list[str]:
        # シンプルな tokenization: 句点/空白で分割 + 2 文字以上
        tokens: list[str] = []
        for chunk in text.replace("\n", "。").split("。"):
            for word in chunk.split():
                w = word.strip("、.,")
                if len(w) >= 2:
                    tokens.append(w)
        # 漢字語も強引に混ぜる (文字ベースの simple tokenizer)
        for word in ["田中", "佐藤", "スカラー商事", "東京", "大阪", "プロジェクト"]:
            count = text.count(word)
            for _ in range(count):
                tokens.append(word)
        return tokens

    def save(self, path) -> None:
        # analyzer.json を書き出す (strict_match / compat_match の最小限)
        config = AnalyzerConfig(
            strict_match={
                "model_name": "stub",
                "split_mode": "STUB",
                "pos_allowlist": ["NOUN"],
                "stopwords": [],
                "lemma_rules": "stub",
                "normalization": {"nfkc": True, "lf_only": True, "strip_trailing": True, "collapse_spaces": True},
                "sudachidict_binary_sha256": "stub",
                "model_checksum": "stub",
            },
            compat_match={"ginza": "stub-0.0", "spacy": "stub-0.0", "sudachipy": "stub-0.0"},
            tfidf={"min_df": 1, "max_df": 0.95},
        )
        Path(path).write_text(
            json.dumps(config.to_dict(), ensure_ascii=False), encoding="utf-8"
        )

    def build_config(self) -> AnalyzerConfig:
        return AnalyzerConfig(
            strict_match={"model_name": "stub"},
            compat_match={"ginza": "stub-0.0"},
            tfidf={},
        )


class _ScriptedLLM:
    model_id = "mock@v1"

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate(self, prompt: str, max_tokens: int) -> GenerateResult:
        self.calls += 1
        r = self._responses.pop(0) if self._responses else GenerateResult(
            text="OK", input_tokens=5, output_tokens=5, model_id="mock@v1", finish_reason="end_turn"
        )
        if isinstance(r, Exception):
            raise r
        return r


def _ok(text: str = "OK") -> GenerateResult:
    return GenerateResult(text=text, input_tokens=5, output_tokens=10, model_id="mock@v1", finish_reason="end_turn")


# ---- Happy path ---------------------------------------------------


def _write_sample(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    # 固有名詞密度の高い corpus
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
    (input_dir / "02.txt").write_text(
        "佐藤は最近異動した。"
        "佐藤は新プロジェクトの一員である。"
        "田中と佐藤は協力している。"
        "東京での会議が成功した。\n",
        encoding="utf-8",
    )


def test_ingest_skip_wiki_produces_artifacts(tmp_path: Path) -> None:
    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"

    analyzer = _StubAnalyzer()
    cfg = IngestConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        skip_wiki=True,
        target_chars=30,
        overlap_chars=5,
        max_chunk_chars=200,
    )
    runner = IngestRunner(
        cfg,
        analyzer_factory=lambda path: analyzer,
        llm_factory=lambda backend, config: _ScriptedLLM([]),
    )
    result = runner.run()
    assert result.exit_code == 0, result.errors
    assert (output_dir / "chunks.jsonl").exists()
    assert (output_dir / "vocab.npz").exists()
    assert (output_dir / "analyzer.json").exists()
    assert (output_dir / "log.md").exists()
    # skip_wiki なら entities/manifest.json は作られる (空だが) が、*.md は無い
    assert not list((output_dir / "entities").glob("*.md")) if (output_dir / "entities").exists() else True


def test_ingest_with_wiki_generates_entity_pages(tmp_path: Path) -> None:
    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"

    entity_map = {
        "田中": [EntityMention("田中", "PERSON", 0, 2)],
        "佐藤": [EntityMention("佐藤", "PERSON", 0, 2)],
        "スカラー商事": [EntityMention("スカラー商事", "ORG", 0, 6)],
    }
    analyzer = _StubAnalyzer(entity_map)
    # mention_count の数だけ LLM 呼ばれる + pre-flight 1 回
    llm = _ScriptedLLM([_ok("pre-flight"), _ok("田中要約"), _ok("商事要約"), _ok("佐藤要約")])
    cfg = IngestConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        skip_wiki=False,
        target_chars=25,
        overlap_chars=0,
        max_chunk_chars=200,
        min_mentions=2,
        min_chunks=1,
        llm_backend="anthropic",
    )
    runner = IngestRunner(
        cfg,
        analyzer_factory=lambda path: analyzer,
        llm_factory=lambda backend, config: llm,
    )
    result = runner.run()
    assert result.exit_code == 0, result.errors
    # エンティティ .md が生成されている
    ent_dir = output_dir / "entities"
    md_files = list(ent_dir.glob("*.md"))
    assert len(md_files) >= 1
    # index.md 確認
    index = (output_dir / "index.md").read_text("utf-8")
    assert "Entity Index" in index
    # log.md 確認
    log = (output_dir / "log.md").read_text("utf-8")
    assert "chunks generated:" in log
    assert "wiki succeeded:" in log


def test_ingest_fails_on_empty_input_dir(tmp_path: Path) -> None:
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
    assert result.exit_code != 0
    assert any("no .txt" in e for e in result.errors)


def test_ingest_skips_empty_and_bad_utf8_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "samples"
    input_dir.mkdir()
    (input_dir / "01.txt").write_text("", encoding="utf-8")  # 空ファイル
    (input_dir / "02.txt").write_bytes(b"\xff\xfe\x00 invalid")  # 不正 UTF-8
    (input_dir / "03.txt").write_text(
        "田中はスカラー商事で働いています。東京に住んでいます。",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    cfg = IngestConfig(
        input_dir=input_dir, output_dir=output_dir, skip_wiki=True, target_chars=20,
        max_chunk_chars=100,
    )
    runner = IngestRunner(
        cfg,
        analyzer_factory=lambda path: _StubAnalyzer(),
        llm_factory=lambda backend, config: _ScriptedLLM([]),
    )
    result = runner.run()
    assert result.exit_code == 0, result.errors
    assert result.skipped_files == 2
    assert any("empty file" in w for w in result.warnings)
    assert any("utf-8 decode failed" in w for w in result.warnings)


def test_ingest_openai_backend_fails_fast(tmp_path: Path) -> None:
    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"

    cfg = IngestConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        skip_wiki=False,
        llm_backend="openai",
    )

    # 実物の get_client を使う (OpenAIPlaceholderClient がコンストラクタで raise する)
    from lorebook_chunker.llm import get_client
    runner = IngestRunner(
        cfg,
        analyzer_factory=lambda path: _StubAnalyzer(),
        llm_factory=get_client,
    )
    result = runner.run()
    assert result.exit_code != 0
    assert any("OpenAI" in e or "v2" in e for e in result.errors)


def test_ingest_atomic_swap_replaces_existing(tmp_path: Path) -> None:
    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "old_file.txt").write_text("これは前回のゴミ", encoding="utf-8")

    cfg = IngestConfig(
        input_dir=input_dir, output_dir=output_dir, skip_wiki=True, target_chars=30, max_chunk_chars=200,
    )
    runner = IngestRunner(
        cfg,
        analyzer_factory=lambda path: _StubAnalyzer(),
        llm_factory=lambda backend, config: _ScriptedLLM([]),
    )
    result = runner.run()
    assert result.exit_code == 0
    # 古いファイルは消えている
    assert not (output_dir / "old_file.txt").exists()
    # 新しいファイルがある
    assert (output_dir / "chunks.jsonl").exists()


def test_run_ingest_cli_wrapper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_ingest(args) が argparse.Namespace を受けて動くこと."""
    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"

    args = argparse.Namespace(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        skip_wiki=True,
        max_llm_calls=None,
        force_regenerate=False,
        retry_failed=False,
        llm_backend="anthropic",
        llm_model=None,
        config=None,
        command="ingest",
    )
    # analyzer_factory を差し替えるため、monkeypatch で default を上書き
    import lorebook_chunker.ingest as mod
    monkeypatch.setattr(mod, "default_analyzer_factory", lambda path: _StubAnalyzer())
    exit_code = run_ingest(args)
    assert exit_code == 0
    assert (output_dir / "chunks.jsonl").exists()


def test_ingest_passes_llm_model_to_factory(tmp_path: Path) -> None:
    """--llm-model が llm_factory に {"model": ...} として渡ること."""
    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"

    captured: list[tuple[str, dict[str, Any] | None]] = []

    def _capturing_factory(backend: str, config: dict[str, Any] | None):
        captured.append((backend, config))
        return _ScriptedLLM([])

    cfg = IngestConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        skip_wiki=False,
        llm_backend="ollama",
        llm_model="qwen3:8b",
    )
    result = IngestRunner(
        cfg,
        analyzer_factory=lambda path: _StubAnalyzer(),
        llm_factory=_capturing_factory,
    ).run()
    assert result.exit_code == 0
    assert captured == [("ollama", {"model": "qwen3:8b"})]


def test_ingest_omits_model_key_when_flag_absent(tmp_path: Path) -> None:
    """--llm-model 未指定時は空 dict を渡してバックエンド既定値を使わせる."""
    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"

    captured: list[tuple[str, dict[str, Any] | None]] = []

    def _capturing_factory(backend: str, config: dict[str, Any] | None):
        captured.append((backend, config))
        return _ScriptedLLM([])

    cfg = IngestConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        skip_wiki=False,
        llm_backend="anthropic",
        llm_model=None,
    )
    IngestRunner(
        cfg,
        analyzer_factory=lambda path: _StubAnalyzer(),
        llm_factory=_capturing_factory,
    ).run()
    assert captured == [("anthropic", {})]


def test_retry_failed_and_force_regenerate_cooccur_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"
    args = argparse.Namespace(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        skip_wiki=True,
        max_llm_calls=None,
        force_regenerate=True,
        retry_failed=True,
        llm_backend="anthropic",
        llm_model=None,
        config=None,
        command="ingest",
    )
    import lorebook_chunker.ingest as mod
    monkeypatch.setattr(mod, "default_analyzer_factory", lambda path: _StubAnalyzer())
    with caplog.at_level("WARNING"):
        run_ingest(args)
    assert any("force-regenerate takes precedence" in rec.message for rec in caplog.records)
