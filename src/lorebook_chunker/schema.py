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


# U4: SkipReport の有効 reason 値 (enum-like 文字列).
# これらは `skipped_files.jsonl` と (U5 以降) `run_report.json` に埋め込まれる
# ため、値は安定した契約の一部として扱う. 新しい値を追加するときはテスト
# `test_skip_report.py::test_skip_report_reason_values_are_stable` を更新する.
SKIP_REASON_EMPTY_FILE = "empty_file"
SKIP_REASON_ENCODING_DECODE_FAILED = "encoding_decode_failed"
SKIP_REASON_ENCODING_DETECTION_FAILED = "encoding_detection_failed"
SKIP_REASON_PERMISSION_DENIED = "permission_denied"
SKIP_REASON_FILE_NOT_FOUND = "file_not_found"

VALID_SKIP_REASONS: frozenset[str] = frozenset(
    {
        SKIP_REASON_EMPTY_FILE,
        SKIP_REASON_ENCODING_DECODE_FAILED,
        SKIP_REASON_ENCODING_DETECTION_FAILED,
        SKIP_REASON_PERMISSION_DENIED,
        SKIP_REASON_FILE_NOT_FOUND,
    }
)


@dataclass
class SkipReport:
    """個別ファイルが ingest パイプラインから除外された理由を構造化した1件.

    ingest.py の読み込みループが warning 文字列と並行して emit する. `log.md`
    への flat 文字列追加は後方互換性のため維持されるが、機械可読な副作用は
    `skipped_files.jsonl` (このクラスを 1 行1つの JSON として dump) 側に集約される.

    有効な `reason` 値は `VALID_SKIP_REASONS` を参照:
      - "empty_file": decode に成功したが strip 後に空
      - "encoding_decode_failed": 明示指定 encoding が bytes を decode できなかった
      - "encoding_detection_failed": auto 検出が曖昧 / 検出器未導入 / サイズ不足
      - "permission_denied": OS 側で read 権限拒否
      - "file_not_found": discovery と read の間でファイルが消えた
    """

    path: str
    reason: str
    detail: str | None = None
    encoding_attempted: str | None = None
    size_bytes: int | None = None

    def to_jsonable(self) -> dict[str, Any]:
        """JSON 可搬な dict を返す. None フィールドはそのまま JSON null になる."""
        return {
            "path": self.path,
            "reason": self.reason,
            "detail": self.detail,
            "encoding_attempted": self.encoding_attempted,
            "size_bytes": self.size_bytes,
        }


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


# U1: AnalyzerVersionMismatchError / AnalyzerNEUnavailableError は
# analyzer 初期化 / load フェーズで raise されるため AnalyzerInitError を継承
# (exit code 4). ChunkFileCorruptError は chunks.jsonl 読み込み時の破損検知
# なので RunReportError ではなく ChunkingError (exit 13) を継承 — chunk
# artifact の整合性問題という semantic に対応し、query/lint 側で
# catch-all-LorebookError dispatch に乗る.
# RuntimeError 多重継承は採用せず (grep `except RuntimeError` で捕捉する
# caller が存在しないことを U1 Verification で確認済み).
from lorebook_chunker.errors import AnalyzerInitError, ChunkingError


class AnalyzerVersionMismatchError(AnalyzerInitError):
    """analyzer.json の strict_match が現在のランタイムと不一致."""


class AnalyzerNEUnavailableError(AnalyzerInitError):
    """Ginza の `token._.ne` 拡張が populate されていない."""


class ChunkFileCorruptError(ChunkingError):
    """chunks.jsonl の行が JSON として壊れている."""
