"""データクラス: analyzer config / chunk record / entity structures.

すべて JSON シリアライズ可能 (dataclasses.asdict 経由)。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict


@dataclass(frozen=True)
class EntityMention:
    """1 チャンク内の 1 回の NER ヒット."""

    name: str
    ner_label: str
    char_start: int
    char_end: int


class KeywordEntry(TypedDict):
    """TF-IDF 上位キーワードの 1 件."""

    term: str
    tfidf: float


class NormalizationDict(TypedDict):
    """analyzer.json strict_match.normalization の内訳."""

    nfkc: bool
    lf_only: bool
    strip_trailing: bool
    collapse_spaces: bool


class StrictMatchDict(TypedDict):
    """analyzer.json strict_match: 保存時と load 時で byte-for-byte 一致が必要な 8 フィールド."""

    model_name: str
    model_checksum: str
    split_mode: str
    pos_allowlist: list[str]
    stopwords: list[str]
    lemma_rules: str
    sudachidict_binary_sha256: str
    normalization: NormalizationDict


class CompatMatchDict(TypedDict):
    """analyzer.json compat_match: major.minor 一致で足りる 5 フィールド."""

    ginza: str
    spacy: str
    sudachipy: str
    sudachidict_package: str
    sudachidict_package_version: str


@dataclass
class ChunkRecord:
    """chunks.jsonl 1 行分."""

    chunk_id: str
    row_index: int
    source: str
    char_start: int
    char_end: int
    text: str
    top_keywords: list[KeywordEntry] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EntityAggregate:
    """コーパス横断のエンティティ集約."""

    name: str
    ner_label: str
    mention_count: int
    chunk_count: int
    chunk_ids: list[str]
    cooccurring_entities: list[dict[str, str]] = field(default_factory=list)


@dataclass
class AnalyzerConfig:
    """analyzer.json の内容 (strict_match / compat_match / tfidf の3グループ).

    load 時は strict_match 全一致必須、compat_match は major.minor 一致。
    """

    strict_match: dict[str, Any]
    compat_match: dict[str, Any]
    tfidf: dict[str, Any]
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "strict_match": self.strict_match,
            "compat_match": self.compat_match,
            "tfidf": self.tfidf,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnalyzerConfig":
        return cls(
            version=d.get("version", 1),
            strict_match=d["strict_match"],
            compat_match=d["compat_match"],
            tfidf=d.get("tfidf", {}),
        )


class AnalyzerVersionMismatchError(RuntimeError):
    """analyzer.json の strict_match が現在のランタイムと不一致."""


class AnalyzerNEUnavailableError(RuntimeError):
    """Ginza の `token._.ne` 拡張が populate されていない."""


class ChunkFileCorruptError(RuntimeError):
    """chunks.jsonl の行が JSON として壊れている."""
