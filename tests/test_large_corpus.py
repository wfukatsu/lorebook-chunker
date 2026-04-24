"""U8: 大容量 corpus 統合テスト (env-gated by `RUN_LARGE=1`).

既定の `pytest tests/` では全 skip. `RUN_LARGE=1 pytest tests/test_large_corpus.py`
でローカル実行.

テスト対象:
- 100 files × 5 KB → 60秒以内に ingest 完走 (stub analyzer + skip_wiki)
- 同一 corpus を 2 回 ingest → entities_generated が一致 (決定論)
- 1 file × 500 KB → sudachi byte-limit split path を cover (クラッシュしない)

`generate_large_corpus` は `seed=42` で決定論. 詳細は
`tests/fixtures/large_corpus_generator.py`.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from lorebook_chunker.ingest import IngestConfig, IngestRunner
from lorebook_chunker.schema import EntityMention
from tests.conftest import _ScriptedLLM, _StubAnalyzer
from tests.fixtures.large_corpus_generator import generate_large_corpus


pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_LARGE"),
    reason="large corpus tests require RUN_LARGE=1",
)


# ---- helpers --------------------------------------------------------


def _stub_analyzer() -> _StubAnalyzer:
    """generator が埋め込むトークンに対応した stub."""
    return _StubAnalyzer(
        {
            "田中": [EntityMention("田中", "PERSON", 0, 2)],
            "佐藤": [EntityMention("佐藤", "PERSON", 0, 2)],
            "スカラー商事": [EntityMention("スカラー商事", "ORG", 0, 6)],
            "東京支社": [EntityMention("東京支社", "ORG", 0, 4)],
            "大阪支社": [EntityMention("大阪支社", "ORG", 0, 4)],
        }
    )


def _run(
    input_dir: Path,
    output_dir: Path,
    *,
    skip_wiki: bool = True,
    **overrides,
):
    cfg = IngestConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        skip_wiki=skip_wiki,
        target_chars=300,
        max_chunk_chars=1500,
        min_mentions=2,
        min_chunks=1,
        show_progress=False,
        recursive=False,
        **overrides,
    )
    runner = IngestRunner(
        cfg,
        analyzer_factory=lambda path: _stub_analyzer(),
        llm_factory=lambda backend, config: _ScriptedLLM([]),
    )
    return runner.run()


# ---- Tests ----------------------------------------------------------


def test_100_files_ingests_in_under_60_seconds(tmp_path: Path) -> None:
    """100 files × 5 KB (計 ~500 KB) を stub pipeline で 60 秒以内に ingest 完走.

    wiki はスキップ (LLM 呼出が wall time を支配するため). 主な検査対象は
    chunking + tfidf + entity aggregation の純パイプラインスループット.
    """
    input_dir = tmp_path / "samples"
    generate_large_corpus(input_dir, n_files=100, avg_size_kb=5, seed=42)
    output_dir = tmp_path / "out"

    t0 = time.perf_counter()
    result = _run(input_dir, output_dir, skip_wiki=True)
    elapsed = time.perf_counter() - t0

    assert result.exit_code == 0, result.errors
    assert result.total_input_files == 100
    assert result.chunks_generated > 0
    assert elapsed < 60.0, (
        f"100 files / 5KB ingest took {elapsed:.2f}s, exceeds 60s budget"
    )


def test_same_corpus_ingested_twice_yields_identical_entity_count(
    tmp_path: Path,
) -> None:
    """同一 seed で生成した corpus を 2 つの独立 output dir に ingest.

    entities_generated が同数なら NER aggregation が決定論的に働いている.
    corpus 自体の determinism は generator 側で保証している (seed=42).
    """
    input_dir = tmp_path / "samples"
    generate_large_corpus(input_dir, n_files=100, avg_size_kb=5, seed=42)

    r1 = _run(input_dir, tmp_path / "out1", skip_wiki=True)
    r2 = _run(input_dir, tmp_path / "out2", skip_wiki=True)
    assert r1.exit_code == 0
    assert r2.exit_code == 0
    assert r1.entities_generated == r2.entities_generated, (
        f"entities_generated differs across runs: {r1.entities_generated} vs "
        f"{r2.entities_generated} (non-deterministic pipeline?)"
    )
    assert r1.chunks_generated == r2.chunks_generated


def test_single_500kb_file_triggers_sudachi_byte_limit_split(
    tmp_path: Path,
) -> None:
    """1 file × 500 KB は `SUDACHI_INPUT_BYTE_LIMIT=40000` を十分超え、
    `analyzer._split_into_nlp_segments` のハードカット経路を通過する.

    stub analyzer は sudachi を使わないため本テストは主に
    "500 KB 単一ファイルでも pipeline が crash せず chunks を出す" ことを検査.
    実 GiNZA 経路での byte-limit split 詳細は U11 (実 samples) で cover.
    """
    input_dir = tmp_path / "samples"
    generate_large_corpus(input_dir, n_files=1, avg_size_kb=500, seed=42)
    # 500 KB 以上あることを確認 (generator の sanity)
    only = next(input_dir.iterdir())
    size_kb = only.stat().st_size / 1024
    assert size_kb >= 400, f"generator produced {size_kb:.1f} KB, expected >=400"

    output_dir = tmp_path / "out"
    result = _run(input_dir, output_dir, skip_wiki=True)
    assert result.exit_code == 0, result.errors
    assert result.total_input_files == 1
    assert result.chunks_generated > 0, "large single file should yield > 0 chunks"
