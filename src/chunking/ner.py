"""NER extraction + entity aggregation + filename sanitization.

analyzer の iter_entities (token._.ne BIO walk) に依存するが、本モジュール自体は
EntityMention 配列を入力にテストできるよう純粋関数として書かれている。
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

from chunking.schema import ChunkRecord, EntityAggregate, EntityMention

DEFAULT_TARGET_LABELS: tuple[str, ...] = ("PERSON", "ORG", "LOC", "PRODUCT")
DEFAULT_MIN_MENTIONS = 3
DEFAULT_MIN_CHUNKS = 2

# Windows 予約名 (case-insensitive)
_WINDOWS_RESERVED = frozenset(
    {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }
)

_UNSAFE_FILENAME_RE = re.compile(r'[/\\:*?"<>|\s]')
_FILENAME_MAX_BYTES = 128


class SupportsIterEntities(Protocol):
    def iter_entities(self, text: str) -> Iterable[EntityMention]: ...


@dataclass
class AggregationStats:
    """集約のサマリ (lint が消費する)."""

    total_detected_entities: int  # 対象ラベルで検出された生のエンティティ (重複を含まない名前+ラベル数)
    accepted_entities: int  # しきい値通過
    skipped_entities: int  # しきい値未達


def extract_entities_per_chunk(
    analyzer: SupportsIterEntities,
    chunks: Sequence[ChunkRecord],
    target_labels: Iterable[str] = DEFAULT_TARGET_LABELS,
) -> list[list[EntityMention]]:
    """各 ChunkRecord の text に対して NER を実行. 対象ラベル外は除外."""
    labels = set(target_labels)
    out: list[list[EntityMention]] = []
    for chunk in chunks:
        mentions = [m for m in analyzer.iter_entities(chunk.text) if m.ner_label in labels]
        out.append(mentions)
    return out


def attach_entities_to_chunks(
    chunks: list[ChunkRecord],
    entities_per_chunk: list[list[EntityMention]],
) -> None:
    """各 ChunkRecord の entities フィールドに {name, ner_label, char_start, char_end} を格納.

    char offset はチャンク内相対位置。in-place 更新。
    """
    for chunk, mentions in zip(chunks, entities_per_chunk):
        chunk.entities = [
            {
                "name": m.name,
                "ner_label": m.ner_label,
                "char_start": m.char_start,
                "char_end": m.char_end,
            }
            for m in mentions
        ]


def aggregate_entities(
    chunks: Sequence[ChunkRecord],
    entities_per_chunk: Sequence[Sequence[EntityMention]],
    *,
    min_mentions: int = DEFAULT_MIN_MENTIONS,
    min_chunks: int = DEFAULT_MIN_CHUNKS,
) -> tuple[list[EntityAggregate], AggregationStats]:
    """コーパス横断で (name, ner_label) 単位に集約. しきい値未達はスキップ."""
    if len(chunks) != len(entities_per_chunk):
        raise ValueError("chunks and entities_per_chunk must have same length")

    per_key: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"mentions": 0, "chunk_ids": set(), "cooccurring": set()}
    )
    for chunk, mentions in zip(chunks, entities_per_chunk):
        chunk_co: set[tuple[str, str]] = {(m.name, m.ner_label) for m in mentions}
        for mention in mentions:
            key = (mention.name, mention.ner_label)
            per_key[key]["mentions"] += 1
            per_key[key]["chunk_ids"].add(chunk.chunk_id)
            per_key[key]["cooccurring"].update(chunk_co - {key})

    total = len(per_key)
    accepted: list[EntityAggregate] = []
    skipped = 0
    for (name, label), d in per_key.items():
        mentions = int(d["mentions"])
        chunk_ids_set: set[str] = d["chunk_ids"]  # type: ignore[assignment]
        if mentions < min_mentions or len(chunk_ids_set) < min_chunks:
            skipped += 1
            continue
        cooccurring = sorted(
            ({"name": n, "ner_label": l} for (n, l) in d["cooccurring"]),  # type: ignore[union-attr]
            key=lambda obj: (obj["ner_label"], obj["name"]),
        )
        # chunk_ids はチャンクの出現順を保つよう、順序付きで抽出
        ordered_chunk_ids = [
            c.chunk_id for c in chunks if c.chunk_id in chunk_ids_set
        ]
        accepted.append(
            EntityAggregate(
                name=name,
                ner_label=label,
                mention_count=mentions,
                chunk_count=len(chunk_ids_set),
                chunk_ids=ordered_chunk_ids,
                cooccurring_entities=cooccurring,
            )
        )
    accepted.sort(key=lambda e: (-e.mention_count, -e.chunk_count, e.name))
    stats = AggregationStats(
        total_detected_entities=total,
        accepted_entities=len(accepted),
        skipped_entities=skipped,
    )
    return accepted, stats


# ---- filename sanitization -----------------------------------------------

def sanitize_entity_filename(ner_label: str, entity_name: str) -> str:
    """`{label}__{sanitized}.md` 形式で衝突安全なファイル名を生成.

    - 不正文字 `/ \\ : * ? " < > |` と任意の空白を `_` に置換
    - Windows 予約名 (CON, NUL, COM1, ...) は末尾に `_` を付与
    - UTF-8 バイト長が 128 を超えたら切り詰めて末尾に short hash を付与
    """
    # label は英字のみ前提 (OntoNotes5) だが、念のため sanitize
    safe_label = _UNSAFE_FILENAME_RE.sub("_", ner_label)
    safe_name = _UNSAFE_FILENAME_RE.sub("_", entity_name)
    if safe_name.upper() in _WINDOWS_RESERVED:
        safe_name = safe_name + "_"
    base = f"{safe_label}__{safe_name}"
    encoded = base.encode("utf-8")
    if len(encoded) > _FILENAME_MAX_BYTES:
        # 128 - 8 (hash) - 1 (_) = 119 bytes ぶんを prefix として使う
        suffix = "_" + hashlib.sha256(base.encode("utf-8")).hexdigest()[:6]
        target_bytes = _FILENAME_MAX_BYTES - len(suffix.encode("utf-8"))
        # 文字境界で切る
        truncated = encoded[:target_bytes].decode("utf-8", errors="ignore")
        base = truncated + suffix
    return base + ".md"
