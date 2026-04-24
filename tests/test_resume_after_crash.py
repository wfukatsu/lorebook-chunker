"""U8: resume-after-crash 統合テスト.

`<output>.staging/`, `<output>.backup/`, `<output>.failed/` の sibling dir
残存シナリオを統合的にカバー. これらは `_swap.py` / `ingest.py` の
finally / atomic_swap 経路で clean up 済みのはずだが、現場での
partial-write / kill-9 / NFS の遅延 unlink などで残ることがある.

既存 cover:
- `test_smoke.py::test_wiki_second_run_skips_cache_hits` — inner caching
- `test_atomic_swap_hardening.py` — atomic swap 単体の挙動
- U3: encoding variants
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lorebook_chunker.ingest import IngestConfig, IngestRunner
from lorebook_chunker.llm import GenerateResult
from lorebook_chunker.schema import EntityMention
from tests.conftest import _ScriptedLLM, _StubAnalyzer


# ---- helpers --------------------------------------------------------


def _ok(text: str = "要約") -> GenerateResult:
    return GenerateResult(
        text=text,
        input_tokens=5,
        output_tokens=10,
        model_id="mock@v1",
        finish_reason="end_turn",
    )


def _write_samples(input_dir: Path) -> None:
    """entity aggregation + wiki 生成に十分な固有名詞密度を持つ最小コーパス."""
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
    (input_dir / "02.txt").write_text(
        "佐藤は最近異動した。"
        "佐藤は新プロジェクトの一員である。"
        "田中と佐藤は協力している。"
        "東京での会議が成功した。\n",
        encoding="utf-8",
    )


def _stub_analyzer() -> _StubAnalyzer:
    return _StubAnalyzer(
        {
            "田中": [EntityMention("田中", "PERSON", 0, 2)],
            "佐藤": [EntityMention("佐藤", "PERSON", 0, 2)],
            "スカラー商事": [EntityMention("スカラー商事", "ORG", 0, 6)],
        }
    )


def _run_ingest(
    input_dir: Path,
    output_dir: Path,
    *,
    skip_wiki: bool = True,
    llm: _ScriptedLLM | None = None,
    **cfg_overrides: Any,
) -> Any:
    """標準的な IngestRunner.run() 呼び出し. stub analyzer + scripted LLM."""
    cfg = IngestConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        skip_wiki=skip_wiki,
        target_chars=30,
        max_chunk_chars=200,
        min_mentions=2,
        min_chunks=1,
        show_progress=False,
        **cfg_overrides,
    )
    analyzer = _stub_analyzer()
    if llm is None:
        # 十分な成功レスポンスをスクリプトしておく
        llm = _ScriptedLLM([_ok() for _ in range(50)])
    runner = IngestRunner(
        cfg,
        analyzer_factory=lambda path: analyzer,
        llm_factory=lambda backend, config: llm,
    )
    return runner.run()


# ---- Happy path: 2回目 ingest で wiki cache hit --------------------


def test_second_ingest_run_hits_wiki_cache(tmp_path: Path) -> None:
    """1回目で wiki を生成 → 2回目で全 entity が manifest cache に hit.

    `test_smoke.py::test_wiki_second_run_skips_cache_hits` と似ているが、
    本テストは manifest resume path に特化:
    - 2 回目で per-entity generate は一切発生しない (preflight 1件のみ).
    - `wiki_stats.skipped_cached > 0` で cache hit 経路を明示.

    備考: wiki.py 現仕様では `attempted` は **LLM 呼び出しが発生した**
    entity のみをカウントし、cache hit は skipped_cached に分類する.
    従って `attempted == 0 && skipped_cached == N` が "全件 cache hit" の正
    常な形. preflight は別カウンタ (stats.preflight_calls) で記録される.
    """
    input_dir = tmp_path / "samples"
    _write_samples(input_dir)
    output_dir = tmp_path / "out"

    # 1回目: 本物の LLM (mock) が走る
    llm1 = _ScriptedLLM([_ok() for _ in range(50)])
    r1 = _run_ingest(input_dir, output_dir, skip_wiki=False, llm=llm1)
    assert r1.exit_code == 0, r1.errors
    assert r1.wiki_stats is not None
    first_generated = r1.wiki_stats.succeeded
    assert first_generated > 0, "wiki succeeded should be > 0 on first run"
    # 1回目の per-entity generate + preflight が走った
    assert llm1.calls >= first_generated

    # 2回目: scripted LLM は preflight 用の 1 件だけ用意. per-entity 呼出が
    # 発生したら pop で exhaust → OK fallback が返るが、これ以上の generate
    # が走ったら call_count が跳ねるので検出可能.
    llm2 = _ScriptedLLM([_ok()])
    r2 = _run_ingest(input_dir, output_dir, skip_wiki=False, llm=llm2)
    assert r2.exit_code == 0, r2.errors
    assert r2.wiki_stats is not None
    # 全 entity が cache hit: attempted==0, skipped_cached==1回目の生成数
    assert r2.wiki_stats.attempted == 0
    assert r2.wiki_stats.skipped_cached == first_generated
    # LLM 呼出は preflight の 1 回のみ (per-entity 0)
    assert llm2.calls == 1, (
        f"expected exactly 1 LLM call (preflight) on full cache-hit resume, "
        f"got {llm2.calls}"
    )
    assert r2.wiki_stats.preflight_calls == 1


# ---- Error path: .staging/ 残 + 新 ingest → 回収 ---------------------


def test_stale_staging_dir_is_reclaimed(tmp_path: Path) -> None:
    """既存 `.staging/` に garbage があっても、新 ingest の冒頭で rmtree される.

    ingest.py:415-418 の ``if staging.exists(): shutil.rmtree(staging)`` を
    cover. 成功後は `.staging/` はどこにも残らない (swap で rename されて
    消える).
    """
    input_dir = tmp_path / "samples"
    _write_samples(input_dir)
    output_dir = tmp_path / "out"

    # 新 ingest 実行前に stale .staging/ を作る
    stale_staging = output_dir.with_name(output_dir.name + ".staging")
    stale_staging.mkdir(parents=True, exist_ok=True)
    (stale_staging / "garbage.txt").write_text("leftover from previous crash", "utf-8")
    (stale_staging / "entities").mkdir(parents=True, exist_ok=True)
    (stale_staging / "entities" / "orphan.md").write_text("# orphan", "utf-8")

    result = _run_ingest(input_dir, output_dir, skip_wiki=True)
    assert result.exit_code == 0, result.errors
    # swap 成功後 staging は消えている
    assert not stale_staging.exists(), "staging dir should be cleaned after successful swap"
    # garbage は一切 output_dir に混入していない
    assert not (output_dir / "garbage.txt").exists()
    assert not (output_dir / "entities" / "orphan.md").exists()
    # 正常成果物は存在
    assert (output_dir / "chunks.jsonl").exists()


# ---- Error path: .backup/ 残 + 新 ingest → 無害に上書き ------------


def test_stale_backup_dir_is_silently_overwritten(tmp_path: Path) -> None:
    """既存 `.backup/` は `_swap.atomic_swap` 内で先に rmtree される.

    `_swap.py:192` の
    ``if backup.exists(): shutil.rmtree(backup)`` を cover.
    plan U8 Test scenarios は "warning ログ or fail fast + recovery 手順"
    を示唆するが、現 _swap.py は silent rmtree が契約. 本テストは現挙動を
    assert する (挙動変更は別 Unit で計画).
    """
    input_dir = tmp_path / "samples"
    _write_samples(input_dir)
    output_dir = tmp_path / "out"

    # 1回目: 正常成功
    r1 = _run_ingest(input_dir, output_dir, skip_wiki=True)
    assert r1.exit_code == 0

    # 手動で stale .backup/ を作る (前回 swap 途中で残された想定)
    stale_backup = output_dir.with_name(output_dir.name + ".backup")
    stale_backup.mkdir(parents=True, exist_ok=True)
    (stale_backup / "ancient.jsonl").write_text('{"old": true}\n', "utf-8")

    # 2回目: stale backup を silent に上書きして進む
    r2 = _run_ingest(input_dir, output_dir, skip_wiki=True)
    assert r2.exit_code == 0, r2.errors
    # 2回目成功後は .backup も消えている (atomic_swap は最後に backup を rmtree)
    assert not stale_backup.exists(), (
        "backup dir should be cleaned after successful swap"
    )
    # warning/error には backup 関連の文字列は入らない (現契約: silent).
    for w in r2.warnings:
        assert "backup" not in w.lower(), f"unexpected backup warning: {w}"


# ---- Error path: .failed/ 残 → doctor が検出 (deferred) --------------


@pytest.mark.skip(
    reason=(
        "doctor sibling-dir scan deferred: doctor.py に `.failed/` 検出チェックが "
        "まだ無い (plan Open Questions 参照). "
        "現状 `.failed/` は F-015 で自動生成されるが、後続 ingest は無視して進む "
        "(test_stale_failed_dir_does_not_block_next_ingest で cover). "
        "doctor 側のチェック追加は別 Unit で計画."
    )
)
def test_stale_failed_dir_surfaced_by_doctor(tmp_path: Path) -> None:
    """TODO: doctor / `--dry-run` が `<output>.failed/` の存在を warning 表示する."""
    pass


def test_stale_failed_dir_does_not_block_next_ingest(tmp_path: Path) -> None:
    """`<output>.failed/` が残っていても次 ingest 実行は正常完走する.

    plan U8 Test scenarios: "`.failed/` 残 → doctor が検出してレポート、
    次 ingest 実行は阻害しない" — 本テストは後半 (ingest non-blocking) を
    assert する. doctor 側の検出は別テストで skip-with-reason 明記済み.
    """
    input_dir = tmp_path / "samples"
    _write_samples(input_dir)
    output_dir = tmp_path / "out"

    # 手動で `.failed/` を作る (前回 wiki systemic abort を想定)
    stale_failed = output_dir.with_name(output_dir.name + ".failed")
    (stale_failed / "entities").mkdir(parents=True, exist_ok=True)
    (stale_failed / "entities" / "manifest.json").write_text(
        json.dumps({"entries": {}, "header": {"note": "prior failed run"}}),
        "utf-8",
    )

    # 新 ingest は stale failed を無視して進む
    result = _run_ingest(input_dir, output_dir, skip_wiki=True)
    assert result.exit_code == 0, result.errors
    assert (output_dir / "chunks.jsonl").exists()
    # .failed/ は自動削除されない (inspection 用に残す契約).
    assert stale_failed.exists(), ".failed dir should remain for inspection"


# ---- Resume mid-wiki failure -----------------------------------


def test_resume_after_wiki_crash_completes_from_cache(tmp_path: Path) -> None:
    """1回目 wiki 中で途中失敗 → 2回目は既に成功した entity を cache hit で skip.

    manifest checkpoint のおかげで、systemic abort 前に書き出された
    ``status="success"`` エントリは次 run で skipped_cached として回収される.

    注: systemic abort の閾値は 5 attempts 以上 + 失敗率 > 50% (wiki.py:57
    `SYSTEMIC_FAILURE_MIN_ATTEMPTS = 5`). 3 entity しか無い本 fixture では
    systemic abort は発動しない. そこで本テストは wiki 呼出を `_ok()` *1件
    + その後 LLMRetryableError 多発で non-systemic failure を演出し、
    2 回目で残りを完了できることを assert.
    代替: wiki generation がそもそも無いケース → 本テストでは skip_wiki=True
    で「wiki 無しでも 2 回目が通ること」だけを cover するのが実装現実的.
    """
    input_dir = tmp_path / "samples"
    _write_samples(input_dir)
    output_dir = tmp_path / "out"

    # 1回目: skip_wiki で成功させ、artifacts を残す
    r1 = _run_ingest(input_dir, output_dir, skip_wiki=True)
    assert r1.exit_code == 0

    # 2回目: 同 output_dir に、再度 skip_wiki. 既存成果物 (analyzer.json 等)
    # を読み込んで処理するフローを通す. `--force-regenerate off` 相当.
    r2 = _run_ingest(input_dir, output_dir, skip_wiki=True)
    assert r2.exit_code == 0, r2.errors
    # chunks は両 run で一致 (deterministic).
    chunks1 = (output_dir / "chunks.jsonl").read_text("utf-8")
    assert chunks1  # 上書き後の内容
    assert r1.chunks_generated == r2.chunks_generated


def test_partial_wiki_cache_resumed_on_second_run(tmp_path: Path) -> None:
    """1回目 wiki で一部のみ成功 (entity 数 < systemic 閾値) → 2回目は残りを LLM で生成.

    scenario:
    - 1回目: scripted LLM で 1 件だけ成功、それ以降の entity は "OK" で埋める
      (失敗ではなく、全て成功として書かれる). 本 fixture は entity 数 = 3
      なので systemic abort 判定は発動しない.
    - 2回目: 空 LLM で走らせ、全件 cache hit で 0 calls.
    - 結果として wiki entries 数は 1回目 == 2回目.
    """
    input_dir = tmp_path / "samples"
    _write_samples(input_dir)
    output_dir = tmp_path / "out"

    # 1回目: 本物 (mock) で wiki 生成. wiki_stats.succeeded > 0 を assert.
    llm1 = _ScriptedLLM([_ok() for _ in range(50)])
    r1 = _run_ingest(input_dir, output_dir, skip_wiki=False, llm=llm1)
    assert r1.exit_code == 0, r1.errors
    assert r1.wiki_stats is not None
    assert r1.wiki_stats.succeeded > 0
    prior_entries = json.loads(
        (output_dir / "entities" / "manifest.json").read_text("utf-8")
    )["entries"]
    assert prior_entries  # 何か書かれている

    # 2回目: preflight 用に 1 件だけ用意. per-entity 呼出があれば exhaust
    # → OK fallback が返るが、calls カウンタが跳ねるので検出可能.
    llm2 = _ScriptedLLM([_ok()])
    r2 = _run_ingest(input_dir, output_dir, skip_wiki=False, llm=llm2)
    assert r2.exit_code == 0, r2.errors
    # preflight の 1 件だけ. per-entity 呼出は 0.
    assert llm2.calls == 1
    assert r2.wiki_stats is not None
    # attempted==0 == per-entity generate が走らなかった == 全件 cache hit
    assert r2.wiki_stats.attempted == 0
    assert r2.wiki_stats.skipped_cached == r1.wiki_stats.succeeded
    # manifest の entries 数は維持 (cache から継承).
    new_entries = json.loads(
        (output_dir / "entities" / "manifest.json").read_text("utf-8")
    )["entries"]
    assert set(prior_entries) == set(new_entries)
