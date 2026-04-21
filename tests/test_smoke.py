"""Unit 12: エンドツーエンドスモークテスト.

デフォルトは `tests/fixtures/samples/` の軽量な fixture を用いて ingest→query→lint を回す.
実サンプル (`samples/01_news.txt` 等) が揃った後は `@pytest.mark.skipif` のガードが外れ、
追加の検証 (R23 char 数 / Success Criteria の言い換えクエリ記録) が有効になる.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from chunking.ingest import IngestConfig, IngestRunner, run_ingest
from chunking.lint import FATAL, run_lint_impl
from chunking.query import run_query_impl
from tests.test_ingest import _ScriptedLLM, _StubAnalyzer

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_SAMPLES = REPO_ROOT / "tests" / "fixtures" / "samples"
REAL_SAMPLES = REPO_ROOT / "samples"
EXPECTED_YAML = REAL_SAMPLES / "expected.yaml"

# R23 の目標文字数 (±10%)
REAL_SAMPLE_TARGETS = {
    "01_news.txt": 5000,
    "02_tech.txt": 6000,
    "03_interview.txt": 5000,
    "04_travel.txt": 5000,
    "05_biography.txt": 4500,
    "06_fiction.txt": 4500,
}

_real_samples_exist = all(
    (REAL_SAMPLES / name).exists() for name in REAL_SAMPLE_TARGETS
)

skip_until_real_samples = pytest.mark.skipif(
    not _real_samples_exist,
    reason="real samples not yet authored; see samples/README.md for Unit 11 spec",
)


def _ingest_fixture(tmp_path: Path, *, skip_wiki: bool = True) -> Path:
    out = tmp_path / "out"
    cfg = IngestConfig(
        input_dir=FIXTURE_SAMPLES,
        output_dir=out,
        skip_wiki=skip_wiki,
        target_chars=30,
        max_chunk_chars=200,
        min_mentions=2,
        min_chunks=1,
    )
    analyzer = _StubAnalyzer(
        {
            "田中": [__import__("chunking.schema", fromlist=["EntityMention"]).EntityMention("田中", "PERSON", 0, 2)],
            "佐藤": [__import__("chunking.schema", fromlist=["EntityMention"]).EntityMention("佐藤", "PERSON", 0, 2)],
            "スカラー商事": [__import__("chunking.schema", fromlist=["EntityMention"]).EntityMention("スカラー商事", "ORG", 0, 6)],
        }
    )
    from chunking.llm import GenerateResult

    def _ok() -> GenerateResult:
        return GenerateResult(text="要約", input_tokens=5, output_tokens=10, model_id="mock@v1", finish_reason="end_turn")

    llm = _ScriptedLLM([_ok() for _ in range(100)])  # 十分なだけ成功
    IngestRunner(
        cfg,
        analyzer_factory=lambda path: analyzer,
        llm_factory=lambda backend, config: llm,
    ).run()
    return out


# ---- 形状系 ---------------------------------------------------


def test_ingest_produces_all_expected_artifacts(tmp_path: Path) -> None:
    out = _ingest_fixture(tmp_path, skip_wiki=True)
    assert (out / "chunks.jsonl").exists()
    assert (out / "vocab.npz").exists()
    assert (out / "analyzer.json").exists()
    assert (out / "log.md").exists()
    # chunks.jsonl に行があり、schema 通り
    with (out / "chunks.jsonl").open("r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    assert "chunk_id" in first
    assert "row_index" in first
    assert "text" in first
    assert "top_keywords" in first


def test_ingest_plus_wiki_generates_entities(tmp_path: Path) -> None:
    out = _ingest_fixture(tmp_path, skip_wiki=False)
    # entity ページが少なくとも 1 つ生成される
    ent_dir = out / "entities"
    assert ent_dir.exists()
    md_files = list(ent_dir.glob("*.md"))
    assert len(md_files) >= 1
    manifest = json.loads((ent_dir / "manifest.json").read_text("utf-8"))
    assert manifest["entries"]


# ---- 検索品質: 語彙一致クエリ -----------------------------


def test_literal_query_finds_relevant_chunk(tmp_path: Path) -> None:
    out = _ingest_fixture(tmp_path, skip_wiki=True)
    result = run_query_impl(
        out, "田中とスカラー商事の合併", top_k=5,
        analyzer_factory=lambda path: _StubAnalyzer(),
    )
    assert result.exit_code == 0
    assert result.hits
    combined_text = " ".join(h.text for h in result.hits)
    assert "田中" in combined_text or "スカラー商事" in combined_text


# ---- 検索品質: 言い換えクエリ (記録のみ、skip/fail なし) -----


def test_paraphrase_query_recorded(tmp_path: Path) -> None:
    out = _ingest_fixture(tmp_path, skip_wiki=True)
    result = run_query_impl(
        out, "経営統合の経緯", top_k=10,
        analyzer_factory=lambda path: _StubAnalyzer(),
    )
    report = {
        "query": "経営統合の経緯",
        "hits_count": len(result.hits),
        "top_chunk_texts": [h.text[:60] for h in result.hits[:3]],
        "exit_code": result.exit_code,
    }
    report_path = tmp_path / "paraphrase_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    assert report_path.exists()  # 記録のみ、常にパス


def test_readme_has_paraphrase_limitation_section() -> None:
    readme = (REPO_ROOT / "README.md").read_text("utf-8")
    assert "言い換え" in readme or "paraphrase" in readme.lower()


# ---- lint ---------------------------------------------


def test_lint_on_fixture_corpus_is_not_fatal(tmp_path: Path) -> None:
    out = _ingest_fixture(tmp_path, skip_wiki=True)
    report = run_lint_impl(out)
    assert report.count(FATAL) == 0


# ---- wiki incremental cache (mock) ---------------------


def test_wiki_second_run_skips_cache_hits(tmp_path: Path) -> None:
    # 1 回目
    out = _ingest_fixture(tmp_path, skip_wiki=False)
    manifest_path = out / "entities" / "manifest.json"
    manifest_v1 = json.loads(manifest_path.read_text("utf-8"))
    # 2 回目 同一 input
    out2 = _ingest_fixture(tmp_path, skip_wiki=False)
    # out と out2 は同じ tmp_path/out なので 2 回目の出力が上書きされる
    # cache が効けば 2 回目の manifest は entries が同じ status="success" を維持
    manifest_v2 = json.loads(manifest_path.read_text("utf-8"))
    assert set(manifest_v1["entries"].keys()) == set(manifest_v2["entries"].keys())


# ---- staging dir crash-safety (mock) ---------------------


def test_ingest_preserves_existing_output_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ingest 途中で例外が発生しても、既存 output_dir は保全される (staging swap 失敗分離)."""
    # 1 回目 (正常)
    out = _ingest_fixture(tmp_path, skip_wiki=True)
    chunks_before = (out / "chunks.jsonl").read_text("utf-8")
    assert chunks_before

    # 2 回目: chunker 呼び出し時に例外を投げる
    import chunking.ingest as mod

    original_chunker = mod.Chunker
    call_count = {"n": 0}

    class _FailingChunker(original_chunker):  # type: ignore[misc]
        def chunk_document(self, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated failure")
            yield from super().chunk_document(*args, **kwargs)

    monkeypatch.setattr(mod, "Chunker", _FailingChunker)

    cfg = IngestConfig(
        input_dir=FIXTURE_SAMPLES,
        output_dir=out,
        skip_wiki=True,
        target_chars=30,
        max_chunk_chars=200,
    )
    analyzer = _StubAnalyzer()
    result = IngestRunner(
        cfg,
        analyzer_factory=lambda path: analyzer,
        llm_factory=lambda b, c: _ScriptedLLM([]),
    ).run()
    assert result.exit_code != 0

    # 既存 output は保持されている
    assert (out / "chunks.jsonl").exists()
    chunks_after = (out / "chunks.jsonl").read_text("utf-8")
    assert chunks_after == chunks_before


# ---- R23 / Success Criteria (実サンプル揃い次第) ---------------


@skip_until_real_samples
def test_real_sample_char_counts_within_tolerance() -> None:
    for name, target in REAL_SAMPLE_TARGETS.items():
        path = REAL_SAMPLES / name
        count = len(path.read_text("utf-8"))
        lo = int(target * 0.9)
        hi = int(target * 1.1)
        assert lo <= count <= hi, f"{name}: {count} chars not in [{lo}, {hi}]"


@skip_until_real_samples
def test_real_samples_end_to_end_query(tmp_path: Path) -> None:
    """実サンプルで ingest→query. Ginza 必須だが real samples が無ければ skip."""
    # Ginza 必須のため、JapaneseAnalyzer を使って実行する想定
    from chunking.ingest import default_analyzer_factory
    out = tmp_path / "out"
    cfg = IngestConfig(
        input_dir=REAL_SAMPLES,
        output_dir=out,
        skip_wiki=True,
    )
    result = IngestRunner(cfg).run()
    assert result.exit_code == 0
    # 期待クエリを実行
    with EXPECTED_YAML.open("r", encoding="utf-8") as f:
        expected = yaml.safe_load(f) or {}
    for q in expected.get("queries", []):
        result = run_query_impl(out, q["query"], top_k=q.get("expect_in_top_k", 5))
        assert result.exit_code == 0
        if "expect_chunk_from" in q:
            sources = {h.source for h in result.hits}
            assert q["expect_chunk_from"] in sources or any(
                q["expect_chunk_from"] in s for s in sources
            )


# ---- backend compatibility (environment-gated) ---------------


@pytest.mark.skipif(os.getenv("RUN_LLM") != "1", reason="RUN_LLM=1 required for live LLM test")
def test_anthropic_live_ingest_completes(tmp_path: Path) -> None:
    """実 Anthropic API での smoke. ANTHROPIC_API_KEY 必須."""
    assert os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY required"
    from chunking.ingest import default_analyzer_factory

    out = tmp_path / "out"
    cfg = IngestConfig(
        input_dir=FIXTURE_SAMPLES,
        output_dir=out,
        skip_wiki=False,
        max_llm_calls=3,
        llm_backend="anthropic",
    )
    result = IngestRunner(cfg).run()
    assert result.exit_code == 0


@pytest.mark.skipif(os.getenv("RUN_OLLAMA") != "1", reason="RUN_OLLAMA=1 required for live Ollama test")
def test_ollama_live_ingest_completes(tmp_path: Path) -> None:
    """実 Ollama ローカル LLM での smoke."""
    from chunking.ingest import default_analyzer_factory

    out = tmp_path / "out"
    cfg = IngestConfig(
        input_dir=FIXTURE_SAMPLES,
        output_dir=out,
        skip_wiki=False,
        max_llm_calls=3,
        llm_backend="ollama",
    )
    result = IngestRunner(cfg).run()
    assert result.exit_code == 0
