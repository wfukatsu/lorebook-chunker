"""NER extraction + entity aggregation + filename sanitization.

analyzer の iter_entities (token._.ne BIO walk) に依存するが、本モジュール自体は
EntityMention 配列を入力にテストできるよう純粋関数として書かれている。
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

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


@dataclass
class _EntityAccum:
    """aggregate_entities の per-(name,label) 蓄積用 (F-051)."""

    mentions: int = 0
    chunk_ids: set[str] = field(default_factory=set)
    cooccurring: set[tuple[str, str]] = field(default_factory=set)


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

    per_key: dict[tuple[str, str], _EntityAccum] = defaultdict(_EntityAccum)
    for chunk, mentions in zip(chunks, entities_per_chunk):
        chunk_co: set[tuple[str, str]] = {(m.name, m.ner_label) for m in mentions}
        for mention in mentions:
            key = (mention.name, mention.ner_label)
            acc = per_key[key]
            acc.mentions += 1
            acc.chunk_ids.add(chunk.chunk_id)
            acc.cooccurring.update(chunk_co - {key})

    # F-041: チャンク出現順を 1 度だけ辞書化して、各 entity で O(1) lookup に使う.
    chunk_positions = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}

    total = len(per_key)
    accepted: list[EntityAggregate] = []
    skipped = 0
    for (name, label), acc in per_key.items():
        if acc.mentions < min_mentions or len(acc.chunk_ids) < min_chunks:
            skipped += 1
            continue
        cooccurring = sorted(
            ({"name": n, "ner_label": l} for (n, l) in acc.cooccurring),
            key=lambda obj: (obj["ner_label"], obj["name"]),
        )
        # F-041: O(E·C) から O(C log C) へ: chunk_positions で index 比較.
        ordered_chunk_ids = sorted(acc.chunk_ids, key=chunk_positions.__getitem__)
        accepted.append(
            EntityAggregate(
                name=name,
                ner_label=label,
                mention_count=acc.mentions,
                chunk_count=len(acc.chunk_ids),
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

_DOT_ONLY_RE = re.compile(r"^\.+$")


def sanitize_entity_filename(ner_label: str, entity_name: str) -> str:
    """`{label}__{sanitized}.md` 形式で衝突安全なファイル名を生成.

    - 不正文字 `/ \\ : * ? " < > |` と任意の空白を `_` に置換
    - Windows 予約名 (CON, NUL, COM1, ...) は末尾に `_` を付与
    - UTF-8 バイト長が 128 を超えたら切り詰めて末尾に short hash を付与
    - F-012: `A/B` と `A\\B` が同じ sanitized 形になる衝突を避けるため、
      元の entity_name から導出した短い sha256 prefix を常に suffix に付与する.
    - F-024: 空 / `...` / 先頭ドットなど危険な basename を reject (ValueError).
    """
    # label は英字のみ前提 (OntoNotes5) だが、念のため sanitize
    safe_label = _UNSAFE_FILENAME_RE.sub("_", ner_label)
    safe_name = _UNSAFE_FILENAME_RE.sub("_", entity_name)
    if safe_name.upper() in _WINDOWS_RESERVED:
        safe_name = safe_name + "_"
    # F-012: 元 entity_name の sha256 6-byte prefix を常に付与し、非可逆サニタイズの
    # 衝突を一意化する. label は独立スコープなので label 内ではこれで衝突ゼロ.
    disambiguator = hashlib.sha256(
        f"{ner_label}\x00{entity_name}".encode("utf-8")
    ).hexdigest()[:6]
    base = f"{safe_label}__{safe_name}_{disambiguator}"

    # F-024: 安全性チェック
    stem_after_label = safe_name + "_" + disambiguator
    if (
        not stem_after_label
        or _DOT_ONLY_RE.match(stem_after_label)
        or stem_after_label.startswith(".")
    ):
        raise ValueError(
            f"invalid entity filename stem derived from {entity_name!r}"
        )

    encoded = base.encode("utf-8")
    if len(encoded) > _FILENAME_MAX_BYTES:
        # 128 - 8 (hash) - 1 (_) = 119 bytes ぶんを prefix として使う
        suffix = "_" + hashlib.sha256(base.encode("utf-8")).hexdigest()[:6]
        target_bytes = _FILENAME_MAX_BYTES - len(suffix.encode("utf-8"))
        # 文字境界で切る
        truncated = encoded[:target_bytes].decode("utf-8", errors="ignore")
        base = truncated + suffix
    return base + ".md"
