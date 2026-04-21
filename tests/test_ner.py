"""NER aggregation + filename sanitization のテスト."""
from __future__ import annotations

from lorebook_chunker.ner import (
    DEFAULT_TARGET_LABELS,
    aggregate_entities,
    attach_entities_to_chunks,
    extract_entities_per_chunk,
    sanitize_entity_filename,
)
from lorebook_chunker.schema import ChunkRecord, EntityMention


class _StubAnalyzer:
    """1 チャンクにつき固定のエンティティリストを返す test double."""

    def __init__(self, per_text_entities: dict[str, list[EntityMention]]) -> None:
        self._map = per_text_entities

    def iter_entities(self, text: str):
        return iter(self._map.get(text, []))


def _chunk(chunk_id: str, text: str, row: int = 0) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        row_index=row,
        source="a.txt",
        char_start=0,
        char_end=len(text),
        text=text,
    )


def test_extract_filters_to_target_labels() -> None:
    analyzer = _StubAnalyzer(
        {
            "t1": [
                EntityMention(name="田中", ner_label="PERSON", char_start=0, char_end=2),
                EntityMention(name="2026", ner_label="DATE", char_start=3, char_end=7),
            ],
        }
    )
    chunks = [_chunk("c1", "t1")]
    out = extract_entities_per_chunk(analyzer, chunks, target_labels=("PERSON",))
    assert len(out) == 1
    assert [m.ner_label for m in out[0]] == ["PERSON"]


def test_attach_entities_updates_chunk_records() -> None:
    mention = EntityMention(name="田中", ner_label="PERSON", char_start=0, char_end=2)
    chunks = [_chunk("c1", "田中氏")]
    attach_entities_to_chunks(chunks, [[mention]])
    assert chunks[0].entities == [
        {"name": "田中", "ner_label": "PERSON", "char_start": 0, "char_end": 2}
    ]


def test_aggregate_applies_min_mentions_threshold() -> None:
    chunks = [_chunk(f"c{i}", f"t{i}", row=i) for i in range(4)]
    entities = [
        [EntityMention("田中", "PERSON", 0, 2)],  # c0
        [EntityMention("田中", "PERSON", 0, 2)],  # c1
        [EntityMention("佐藤", "PERSON", 0, 2)],  # c2
        [EntityMention("田中", "PERSON", 0, 2)],  # c3
    ]
    accepted, stats = aggregate_entities(chunks, entities, min_mentions=3, min_chunks=2)
    names = {e.name for e in accepted}
    assert "田中" in names
    assert "佐藤" not in names  # 1 mention のみ
    assert stats.accepted_entities == 1
    assert stats.skipped_entities == 1
    # 「田中」の chunk_ids が出現順で [c0, c1, c3]
    tanaka = next(e for e in accepted if e.name == "田中")
    assert tanaka.chunk_ids == ["c0", "c1", "c3"]
    assert tanaka.mention_count == 3
    assert tanaka.chunk_count == 3


def test_aggregate_applies_min_chunks_threshold() -> None:
    chunks = [_chunk("c0", "t0"), _chunk("c1", "t1")]
    entities = [
        # 同じチャンクで 3 回出現 → mentions=3 だが chunks=1
        [
            EntityMention("田中", "PERSON", 0, 2),
            EntityMention("田中", "PERSON", 10, 12),
            EntityMention("田中", "PERSON", 20, 22),
        ],
        [],
    ]
    accepted, stats = aggregate_entities(chunks, entities, min_mentions=3, min_chunks=2)
    assert len(accepted) == 0
    assert stats.skipped_entities == 1


def test_aggregate_records_cooccurring_entities() -> None:
    chunks = [_chunk(f"c{i}", f"t{i}", row=i) for i in range(3)]
    entities = [
        [
            EntityMention("田中", "PERSON", 0, 2),
            EntityMention("スカラー商事", "ORG", 10, 16),
        ],
        [
            EntityMention("田中", "PERSON", 0, 2),
            EntityMention("スカラー商事", "ORG", 10, 16),
        ],
        [
            EntityMention("田中", "PERSON", 0, 2),
            EntityMention("スカラー商事", "ORG", 10, 16),
        ],
    ]
    accepted, _ = aggregate_entities(chunks, entities, min_mentions=2, min_chunks=2)
    tanaka = next(e for e in accepted if e.name == "田中")
    assert {"name": "スカラー商事", "ner_label": "ORG"} in tanaka.cooccurring_entities
    org = next(e for e in accepted if e.name == "スカラー商事")
    assert {"name": "田中", "ner_label": "PERSON"} in org.cooccurring_entities


def test_aggregate_empty_inputs() -> None:
    accepted, stats = aggregate_entities([], [])
    assert accepted == []
    assert stats.total_detected_entities == 0


def test_sanitize_entity_filename_basic() -> None:
    # F-012 で短い sha256 prefix が suffix として常時付与される
    result = sanitize_entity_filename("PERSON", "田中太郎")
    assert result.startswith("PERSON__田中太郎_")
    assert result.endswith(".md")
    # 決定論的
    assert sanitize_entity_filename("PERSON", "田中太郎") == result


def test_sanitize_entity_filename_handles_unsafe_chars() -> None:
    name = "株式会社A/B"
    result = sanitize_entity_filename("ORG", name)
    assert "/" not in result
    assert result.endswith(".md")
    assert result.startswith("ORG__")


def test_sanitize_entity_filename_disambiguates_slash_vs_backslash() -> None:
    """F-012: `A/B` と `A\\B` は sanitize 後同じ文字列になるが、sha256 suffix で区別される."""
    slash = sanitize_entity_filename("ORG", "A/B")
    backslash = sanitize_entity_filename("ORG", "A\\B")
    assert slash != backslash


def test_sanitize_entity_filename_windows_reserved() -> None:
    result = sanitize_entity_filename("PERSON", "CON")
    assert result.startswith("PERSON__CON__")  # `_` (Windows suffix) + `_` (prefix sep)
    result_lower = sanitize_entity_filename("PERSON", "nul")
    assert result_lower.startswith("PERSON__nul__")


def test_sanitize_entity_filename_long_name_gets_hash() -> None:
    long_name = "あ" * 100  # > 128 bytes as UTF-8 (3 bytes each = 300 bytes)
    result = sanitize_entity_filename("PERSON", long_name)
    assert len(result.encode("utf-8")) <= 128 + 3  # .md は別途
    # short hash suffix が付いている
    assert result.endswith(".md")
    # 2 回実行して結果が同じ (決定論的)
    assert sanitize_entity_filename("PERSON", long_name) == result


def test_sanitize_entity_filename_whitespace_replaced() -> None:
    result = sanitize_entity_filename("ORG", "A B C")
    assert result.startswith("ORG__A_B_C_")
    assert result.endswith(".md")


def test_sanitize_entity_filename_rejects_dot_only_names() -> None:
    """F-024: ドットのみ/空の entity_name は ValueError を上げる."""
    # "..." → sanitize 後も "...", + `_disambig` suffix で dot-only ではなくなるが、
    # entity_name が空や dot-only の場合、stem が "_{hash}" で問題ないケースもある.
    # 先頭ドットはあり得る: 元 name が "." のとき safe_name=".", stem=".{_}{hash}" → starts_with(".")
    import pytest as _pt
    with _pt.raises(ValueError):
        sanitize_entity_filename("ORG", ".")
    with _pt.raises(ValueError):
        sanitize_entity_filename("ORG", "..")


def test_target_labels_default_matches_plan() -> None:
    assert DEFAULT_TARGET_LABELS == ("PERSON", "ORG", "LOC", "PRODUCT")
