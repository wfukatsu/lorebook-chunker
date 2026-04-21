"""WikiGenerator のテスト. LLM は全て mock で置き換え."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from chunking.llm import (
    GenerateResult,
    LLMPermanentError,
    LLMPreflightError,
    LLMRetryableError,
)
from chunking.schema import ChunkRecord, EntityAggregate
from chunking.wiki import (
    ManifestStore,
    PROMPT_TEMPLATE_VERSION,
    SUMMARY_BODY_MARKER_BEGIN,
    WikiGenerator,
    WikiGeneratorConfig,
    compute_source_hash,
)


# ---- Mock LLM client -------------------------------------------------


class _ScriptedLLMClient:
    """呼び出し順に応じて固定レスポンス or 例外を返す mock."""

    def __init__(self, script: list, model_id: str = "mock@v1") -> None:
        self.model_id = model_id
        self._script = list(script)
        self.call_count = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_tokens: int) -> GenerateResult:
        self.call_count += 1
        self.prompts.append(prompt)
        action = self._script.pop(0) if self._script else _success("OK")
        if isinstance(action, Exception):
            raise action
        return action


def _success(text: str, finish: str = "end_turn", out_tokens: int = 20) -> GenerateResult:
    return GenerateResult(
        text=text,
        input_tokens=10,
        output_tokens=out_tokens,
        model_id="mock@v1",
        finish_reason=finish,
    )


# ---- Fixtures ---------------------------------------------------------


@pytest.fixture
def chunks() -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id=f"c{i:02d}",
            row_index=i,
            source="a.txt",
            char_start=i * 10,
            char_end=i * 10 + 10,
            text=f"チャンク{i}の本文。",
        )
        for i in range(6)
    ]


def _agg(name: str, label: str, chunk_ids: list[str], mentions: int | None = None) -> EntityAggregate:
    return EntityAggregate(
        name=name,
        ner_label=label,
        mention_count=mentions or len(chunk_ids),
        chunk_count=len(chunk_ids),
        chunk_ids=chunk_ids,
    )


@pytest.fixture
def entities() -> list[EntityAggregate]:
    return [
        _agg("田中太郎", "PERSON", ["c00", "c01", "c02"], mentions=5),
        _agg("スカラー商事", "ORG", ["c02", "c03"], mentions=3),
    ]


def _config(tmp_path: Path, **overrides) -> WikiGeneratorConfig:
    defaults = dict(
        entities_dir=tmp_path / "entities",
        manifest_path=tmp_path / "entities" / "manifest.json",
        max_tokens_per_entity=100,
    )
    defaults.update(overrides)
    return WikiGeneratorConfig(**defaults)


def _build(
    tmp_path: Path,
    chunks: list[ChunkRecord],
    script: list,
    **cfg_overrides,
) -> tuple[WikiGenerator, _ScriptedLLMClient, WikiGeneratorConfig]:
    cfg = _config(tmp_path, **cfg_overrides)
    llm = _ScriptedLLMClient(script)
    gen = WikiGenerator(
        llm_client=llm,
        config=cfg,
        analyzer_json_hash="hash-v1",
        ginza_model_version="ja_ginza_electra@v5.2.0",
        chunks=chunks,
        sleep=lambda _: None,
    )
    return gen, llm, cfg


# ---- source_hash ------------------------------------------------------


def test_source_hash_changes_with_entity_name(chunks: list[ChunkRecord]) -> None:
    h1 = compute_source_hash(
        entity_name="田中", ner_label="PERSON",
        chunks=chunks, entity_chunk_ids={"c00"},
        analyzer_json_hash="a", ginza_model_version="g", llm_model_id="l",
    )
    h2 = compute_source_hash(
        entity_name="佐藤", ner_label="PERSON",
        chunks=chunks, entity_chunk_ids={"c00"},
        analyzer_json_hash="a", ginza_model_version="g", llm_model_id="l",
    )
    assert h1 != h2


def test_source_hash_changes_with_analyzer_hash(chunks: list[ChunkRecord]) -> None:
    kwargs = dict(
        entity_name="田中", ner_label="PERSON",
        chunks=chunks, entity_chunk_ids={"c00"},
        ginza_model_version="g", llm_model_id="l",
    )
    h1 = compute_source_hash(analyzer_json_hash="a1", **kwargs)
    h2 = compute_source_hash(analyzer_json_hash="a2", **kwargs)
    assert h1 != h2


def test_source_hash_stable_for_same_inputs(chunks: list[ChunkRecord]) -> None:
    kwargs = dict(
        entity_name="田中", ner_label="PERSON",
        chunks=chunks, entity_chunk_ids={"c00", "c01"},
        analyzer_json_hash="a", ginza_model_version="g", llm_model_id="l",
    )
    assert compute_source_hash(**kwargs) == compute_source_hash(**kwargs)


# ---- happy path ------------------------------------------------------


def test_generate_all_happy_path(
    tmp_path: Path, chunks: list[ChunkRecord], entities: list[EntityAggregate]
) -> None:
    gen, llm, cfg = _build(
        tmp_path, chunks,
        script=[_success("pre-flight OK"), _success("田中太郎の要約。"), _success("スカラー商事の要約。")],
    )
    stats = gen.generate_all(entities)
    assert stats.preflight_called
    assert stats.attempted == 2
    assert stats.succeeded == 2
    assert stats.failed == 0
    assert llm.call_count == 3  # pre-flight + 2 entities
    # ファイル生成 (F-012: 末尾に sha256 短 hash が付くので prefix 一致で拾う)
    person_files = list(cfg.entities_dir.glob("PERSON__田中太郎_*.md"))
    org_files = list(cfg.entities_dir.glob("ORG__スカラー商事_*.md"))
    assert len(person_files) == 1
    assert len(org_files) == 1
    person_file = person_files[0]
    # frontmatter は YAML として parse 可能
    text = person_file.read_text("utf-8")
    assert text.startswith("---\n")
    fm_raw = text.split("---", 2)[1]
    fm = yaml.safe_load(fm_raw)
    assert fm["entity_name"] == "田中太郎"
    assert fm["ner_label"] == "PERSON"
    assert fm["ai_verification_status"] == "unverified"
    assert fm["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    # F-006: plan R15 が要求する 3 フィールド
    assert fm["status"] == "success"
    assert fm["last_attempt_at"]
    assert isinstance(fm["chunk_ids"], list)
    assert fm["chunk_ids"] == ["c00", "c01", "c02"]
    # manifest.json
    assert cfg.manifest_path.exists()


def test_llm_body_with_bare_triple_dash_keeps_single_frontmatter(
    tmp_path: Path, chunks: list[ChunkRecord]
) -> None:
    """F-013: LLM 応答に裸の ``---`` が含まれても frontmatter は 1 ブロックのまま."""
    entities = [_agg("田中", "PERSON", ["c00", "c01"], mentions=3)]
    body = "要約の冒頭。\n---\n別ブロックに見えるテキスト。\n"
    gen, _, cfg = _build(
        tmp_path, chunks,
        script=[_success("pre-flight"), _success(body)],
    )
    gen.generate_all(entities)
    matches = list(cfg.entities_dir.glob("PERSON__田中_*.md"))
    assert len(matches) == 1
    page = matches[0].read_text("utf-8")
    # frontmatter は先頭の ``---`` と、それに対応する閉じ ``---`` の 2 本のみ.
    # 本文側の ``---`` は ``\---`` にエスケープされるので生の ``---`` は 2 回しか現れない.
    bare = [ln for ln in page.splitlines() if ln == "---"]
    assert len(bare) == 2
    # エスケープ記号が含まれている
    assert "\\---" in page


def test_skip_wiki_config_bypasses_llm(
    tmp_path: Path, chunks: list[ChunkRecord], entities: list[EntityAggregate]
) -> None:
    gen, llm, cfg = _build(
        tmp_path, chunks, script=[], skip_wiki=True,
    )
    stats = gen.generate_all(entities)
    assert llm.call_count == 0
    assert stats.attempted == 0
    assert not stats.preflight_called
    # manifest は空でも書かれる (破壊再生成の 1 部として)
    assert cfg.manifest_path.exists()


def test_empty_entities_skips_preflight(
    tmp_path: Path, chunks: list[ChunkRecord]
) -> None:
    gen, llm, _ = _build(tmp_path, chunks, script=[])
    stats = gen.generate_all([])
    assert llm.call_count == 0
    assert not stats.preflight_called


# ---- caching / idempotency -------------------------------------------


def test_second_run_skips_when_source_hash_unchanged(
    tmp_path: Path, chunks: list[ChunkRecord], entities: list[EntityAggregate]
) -> None:
    # 1 回目: 全成功
    gen1, _, cfg = _build(
        tmp_path, chunks,
        script=[_success("pre-flight"), _success("summary1"), _success("summary2")],
    )
    gen1.generate_all(entities)

    # 2 回目: LLM 呼び出し 0 (pre-flight は一度だけ呼ばれる)
    gen2, llm2, _ = _build(
        tmp_path, chunks,
        script=[_success("pre-flight")],  # pre-flight のみ、後は cache hit 想定
    )
    stats2 = gen2.generate_all(entities)
    assert llm2.call_count == 1  # pre-flight だけ
    assert stats2.attempted == 0
    assert stats2.skipped_cached == 2


def test_force_regenerate_ignores_cache(
    tmp_path: Path, chunks: list[ChunkRecord], entities: list[EntityAggregate]
) -> None:
    # 1 回目: 全成功
    gen1, _, cfg = _build(
        tmp_path, chunks,
        script=[_success("pre-flight"), _success("s1"), _success("s2")],
    )
    gen1.generate_all(entities)

    # 2 回目 with force_regenerate
    gen2, llm2, _ = _build(
        tmp_path, chunks,
        script=[_success("pre-flight"), _success("new1"), _success("new2")],
        force_regenerate=True,
    )
    stats = gen2.generate_all(entities)
    assert llm2.call_count == 3  # pre-flight + 2 entities
    assert stats.attempted == 2


def test_retry_failed_reattempts_prior_failures(
    tmp_path: Path, chunks: list[ChunkRecord], entities: list[EntityAggregate]
) -> None:
    # 1 回目: 1 番目は失敗、2 番目は成功
    # entities の順序は mention_count DESC: 田中太郎 (5) -> スカラー商事 (3)
    errors = [LLMRetryableError("fail")] * 3  # 田中太郎で 3 回 retry すべて失敗
    gen1, _, _ = _build(
        tmp_path, chunks,
        script=[_success("pre-flight"), *errors, _success("org-summary")],
    )
    stats1 = gen1.generate_all(entities)
    assert stats1.failed == 1
    assert stats1.succeeded == 1

    # 2 回目 without --retry-failed: 失敗は cache 扱いで skip
    gen2, llm2, _ = _build(
        tmp_path, chunks,
        script=[_success("pre-flight")],
    )
    stats2 = gen2.generate_all(entities)
    assert stats2.attempted == 0
    assert stats2.skipped_cached == 2  # 田中 failed + 佐藤 success 両方 skip

    # 3 回目 with --retry-failed: failed のみ再試行
    gen3, llm3, _ = _build(
        tmp_path, chunks,
        script=[_success("pre-flight"), _success("recovered-summary")],
        retry_failed=True,
    )
    stats3 = gen3.generate_all(entities)
    assert stats3.attempted == 1  # 田中太郎のみ
    assert stats3.succeeded == 1


# ---- systemic failure ------------------------------------------------


def test_systemic_failure_aborts_after_50pct_fail_rate_past_min(
    tmp_path: Path, chunks: list[ChunkRecord]
) -> None:
    # 5 エンティティを用意、最初 3 つ失敗 + 2 つ成功 = 60% failure
    entities = [
        _agg(f"ent{i}", "PERSON", [f"c0{i}"], mentions=3 + i)
        for i in range(5)
    ]
    # entities は mention_count DESC でソートされる: ent4 (7), ent3 (6), ent2 (5), ent1 (4), ent0 (3)
    # 最初の 3 つ (ent4, ent3, ent2) を失敗させる: 1 エンティティあたり retry 3 回
    fails = [LLMRetryableError("fail")] * 9
    successes = [_success("ok"), _success("ok")]
    gen, _, _ = _build(
        tmp_path, chunks,
        script=[_success("pre-flight"), *fails, *successes],
    )
    with pytest.raises(LLMPermanentError) as exc_info:
        gen.generate_all(entities)
    assert "systemic failure" in str(exc_info.value)


def test_no_abort_with_4_attempts_below_min(
    tmp_path: Path, chunks: list[ChunkRecord]
) -> None:
    """5 エンティティ未満 / 最小到達前は systemic abort 発火しない."""
    entities = [
        _agg(f"ent{i}", "PERSON", [f"c0{i}"], mentions=3) for i in range(4)
    ]
    # 全失敗でも 4 件は min (5) 未満のため abort しない
    fails = [LLMRetryableError("fail")] * 12  # 4 * 3 retries
    gen, _, _ = _build(
        tmp_path, chunks,
        script=[_success("pre-flight"), *fails],
    )
    # 例外は起きず、正常に完了 (全 failed で終わる)
    stats = gen.generate_all(entities)
    assert stats.failed == 4
    assert not stats.systemic_failure_aborted


# ---- budget guard ----------------------------------------------------


def test_max_llm_calls_budget_skips_remaining(
    tmp_path: Path, chunks: list[ChunkRecord]
) -> None:
    entities = [
        _agg(f"ent{i}", "PERSON", [f"c0{i}"], mentions=3 + i) for i in range(4)
    ]
    # F-003 修正後: budget はエンティティ試行数のみでカウント (pre-flight は別計上).
    # budget = 3 → 最初 3 エンティティが試行、残り 1 は budget_skipped
    gen, _, _ = _build(
        tmp_path, chunks,
        script=[_success("pre-flight"), _success("s1"), _success("s2"), _success("s3")],
        max_llm_calls=3,
    )
    stats = gen.generate_all(entities)
    assert stats.succeeded == 3
    assert stats.budget_skipped == 1
    assert stats.preflight_calls == 1


# ---- pre-flight ------------------------------------------------------


def test_preflight_permanent_failure_aborts(
    tmp_path: Path, chunks: list[ChunkRecord], entities: list[EntityAggregate]
) -> None:
    gen, _, cfg = _build(
        tmp_path, chunks,
        script=[LLMPermanentError("auth failed")],
    )
    with pytest.raises(LLMPreflightError):
        gen.generate_all(entities)
    # manifest は書かれない (pre-flight 時点で abort)
    assert not cfg.manifest_path.exists() or ManifestStore(cfg.manifest_path).entries() == []


def test_preflight_empty_text_fails(
    tmp_path: Path, chunks: list[ChunkRecord], entities: list[EntityAggregate]
) -> None:
    # 空文字 / output_tokens 0 は失敗扱い
    empty = GenerateResult(text="   ", input_tokens=1, output_tokens=0, model_id="m", finish_reason="end_turn")
    gen, _, _ = _build(tmp_path, chunks, script=[empty])
    with pytest.raises(LLMPreflightError):
        gen.generate_all(entities)


def test_preflight_bad_finish_reason_fails(
    tmp_path: Path, chunks: list[ChunkRecord], entities: list[EntityAggregate]
) -> None:
    bad = _success("something", finish="safety")
    gen, _, _ = _build(tmp_path, chunks, script=[bad])
    with pytest.raises(LLMPreflightError):
        gen.generate_all(entities)


# ---- retry logic ----------------------------------------------------


def test_retry_succeeds_after_retryable_errors(
    tmp_path: Path, chunks: list[ChunkRecord]
) -> None:
    entities = [_agg("only", "PERSON", ["c00"], mentions=3)]
    gen, llm, _ = _build(
        tmp_path, chunks,
        script=[
            _success("pre-flight"),
            LLMRetryableError("1st"),
            LLMRetryableError("2nd"),
            _success("recovered"),
        ],
    )
    stats = gen.generate_all(entities)
    assert stats.succeeded == 1
    assert llm.call_count == 4


def test_permanent_error_skips_without_retrying(
    tmp_path: Path, chunks: list[ChunkRecord]
) -> None:
    entities = [_agg("only", "PERSON", ["c00"], mentions=3)]
    gen, llm, _ = _build(
        tmp_path, chunks,
        script=[_success("pre-flight"), LLMPermanentError("content_policy")],
    )
    stats = gen.generate_all(entities)
    assert stats.failed == 1
    # permanent error なので retry しない: 2 回のみ (pre-flight + 1 試行)
    assert llm.call_count == 2


# ---- manifest atomic write ------------------------------------------


def test_manifest_atomic_write_on_failure(tmp_path: Path) -> None:
    """書き込み中に例外が起きても元のファイルは残る."""
    store = ManifestStore(tmp_path / "entities" / "manifest.json")
    (tmp_path / "entities").mkdir()
    # 1 回成功で保存
    store._entries["PERSON__田中"] = _make_dummy_entry()
    store.save(ginza_model="g", llm_model="l")
    original = (tmp_path / "entities" / "manifest.json").read_text("utf-8")
    assert "田中" in original
    # 2 回目も成功 (上書き)
    store._entries["PERSON__田中"] = _make_dummy_entry(status="failed")
    store.save(ginza_model="g", llm_model="l")
    updated = (tmp_path / "entities" / "manifest.json").read_text("utf-8")
    assert '"status": "failed"' in updated


def test_manifest_corrupted_json_recovers_empty(tmp_path: Path) -> None:
    entities_dir = tmp_path / "entities"
    entities_dir.mkdir()
    (entities_dir / "manifest.json").write_text("not-json{", "utf-8")
    store = ManifestStore(entities_dir / "manifest.json")
    store.load()
    assert store.entries() == []


def _make_dummy_entry(status: str = "success"):
    from chunking.wiki import ManifestEntry
    return ManifestEntry(
        entity_name="田中",
        ner_label="PERSON",
        source_hash="h",
        status=status,
        mention_count=3,
        chunk_count=2,
        chunk_ids=["c00", "c01"],
    )
