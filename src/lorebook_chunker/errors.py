"""Single-base exception hierarchy + unified exit-code table.

U1 で導入. 以前は `cli.py` の epilog と `ingest.py:720-729` の
`INGEST_EXIT_CODES` リテラルが重複していたが、ここを single source に集約する.

`LorebookError` を基底とし、サブクラス毎に `exit_code` class attribute を持つ.
library consumer は `except LorebookError` で一括捕捉でき、CLI boundary は
`e.exit_code` を参照するだけで exit code を決定できる.

ChunkingError naming: 本モジュールでは option (a) を採用 — zero_chunks
(exit 5, 既存契約) と generic chunking failure (exit 13) を別クラス
(`ZeroChunksError` / `ChunkingError`) としてモデリング. ingest.py:424-427 の
「全ファイル processed されたが chunk 0」は独立した semantic site であり、
「chunker が runtime error を raise した」とは性質が異なる.
"""
from __future__ import annotations

from typing import Any


JSONPrimitive = str | int | float | bool | None


def _jsonify(value: Any) -> Any:
    """context value を JSON-serializable 形に正規化する.

    - str / int / float / bool / None はそのまま
    - list / tuple / set は要素を再帰的に正規化したリストに
    - dict はキーを str 化し値を再帰的に正規化した dict に
    - それ以外は str() で degrade
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    return str(value)


class LorebookError(Exception):
    """Base class for all taxonomy-managed errors raised by lorebook-chunker.

    `exit_code` class attribute defaults to 10 (unclassified). Subclasses
    override it to pin stable per-category exit codes.

    Instances capture additional machine-readable context via `**kwargs`
    on `__init__`. Use `to_jsonable()` to emit a dict suitable for
    `json.dumps(..., ensure_ascii=False)` — non-serializable context
    values degrade via `str()` rather than raising.
    """

    exit_code: int = 10

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        # context は shallow copy で保持. 値の JSON 正規化は to_jsonable() 側で
        # 行う (raw 値を持つことで caller が `e.context["path"]` のように
        # Path オブジェクトとして参照できる道を残す).
        self.context: dict[str, Any] = dict(context)

    def to_jsonable(self) -> dict[str, Any]:
        """Return a dict shape {class, message, context} safe for json.dumps."""
        return {
            "class": self.__class__.__name__,
            "message": str(self),
            "context": {k: _jsonify(v) for k, v in self.context.items()},
        }


# ---- Preserved (backward-compatible) exit codes ----------------------
# These exit codes were part of the pre-U1 contract and must stay stable.


class ConfigError(LorebookError):
    """設定が入力を生まない / CLI オプションが矛盾する等の configuration 問題."""

    exit_code = 2


class LLMBackendUnavailableError(LorebookError):
    """preflight 段階で LLM backend が使用不可 (auth 失敗 / backend 未サポート等)."""

    exit_code = 3


class AnalyzerInitError(LorebookError):
    """analyzer (spaCy/Ginza) 初期化失敗."""

    exit_code = 4


class ZeroChunksError(LorebookError):
    """chunking phase が 0 chunk しか生成しなかった (全ファイル skip / 内容空)."""

    exit_code = 5


class WikiGenerationError(LorebookError):
    """wiki 生成が systemic abort (失敗率 > 50%) 等で打ち切られた."""

    exit_code = 6


# ---- New codes (11-17) ---------------------------------------------


class InputError(LorebookError):
    """個別ファイル処理中のエラー (permission_denied / file_not_found 等).

    全ファイルが 0 件のような 'configuration が入力を生まなかった' 問題は
    `ConfigError`. 本クラスは per-file レベル.
    """

    exit_code = 11


class EncodingError(LorebookError):
    """入力ファイルのエンコーディング検出 / decode 失敗."""

    exit_code = 12


class ChunkingError(LorebookError):
    """chunker が runtime failure (zero_chunks 以外)."""

    exit_code = 13


class TfidfError(LorebookError):
    """TF-IDF 構築失敗 (dimension mismatch / fit failure)."""

    exit_code = 14


class EntityAggregationError(LorebookError):
    """NER 集計失敗 / threshold config 不整合."""

    exit_code = 15


class AtomicSwapError(LorebookError):
    """staging → output_dir の atomic swap 失敗 (EXDEV / disk full 等)."""

    exit_code = 16


class RunReportError(LorebookError):
    """run_report.json 書き出し失敗 (U5 で消費)."""

    exit_code = 17


# Fallback for unclassified. LorebookError 自体も exit_code=10 だが、
# runtime では subclass を使う; `LorebookError` を直接 raise しないこと.


EXIT_CODE_MAP: dict[type[LorebookError], int] = {
    ConfigError: ConfigError.exit_code,
    LLMBackendUnavailableError: LLMBackendUnavailableError.exit_code,
    AnalyzerInitError: AnalyzerInitError.exit_code,
    ZeroChunksError: ZeroChunksError.exit_code,
    WikiGenerationError: WikiGenerationError.exit_code,
    InputError: InputError.exit_code,
    EncodingError: EncodingError.exit_code,
    ChunkingError: ChunkingError.exit_code,
    TfidfError: TfidfError.exit_code,
    EntityAggregationError: EntityAggregationError.exit_code,
    AtomicSwapError: AtomicSwapError.exit_code,
    RunReportError: RunReportError.exit_code,
}


# 日本語 1 行 description. describe_exit_codes() で CLI epilog に埋め込む.
_EXIT_CODE_DESCRIPTIONS: dict[int, str] = {
    0: "success",
    2: "設定不整合 (入力ファイル無し など)",
    3: "LLM backend 使用不可 (preflight 失敗)",
    4: "analyzer 初期化失敗",
    5: "chunking が 0 chunk しか生成しなかった",
    6: "wiki 生成が systemic abort で打ち切られた",
    10: "未分類例外 (log / stderr 参照)",
    11: "入力ファイル処理エラー (per-file)",
    12: "エンコーディング検出 / decode 失敗",
    13: "chunking runtime 失敗 (generic)",
    14: "TF-IDF 構築失敗",
    15: "NER 集計失敗",
    16: "atomic swap 失敗",
    17: "run_report.json 書き出し失敗",
}


def describe_exit_codes() -> str:
    """CLI epilog に埋め込むための exit code 一覧テキストを返す.

    Format:
        exit codes:
          0   success
          2   ConfigError  設定不整合 (入力ファイル無し など)
          ...
    """
    lines: list[str] = ["exit codes:"]
    # 0 (success) と 10 (unclassified) は特別扱い: class 名を持たない
    lines.append(f"  0   success")
    # class 毎に昇順で列挙 (code 順)
    ordered: list[tuple[int, str, str]] = []
    for cls, code in EXIT_CODE_MAP.items():
        desc = _EXIT_CODE_DESCRIPTIONS.get(code, "")
        ordered.append((code, cls.__name__, desc))
    ordered.sort(key=lambda t: t[0])
    for code, name, desc in ordered:
        lines.append(f"  {code:<3} {name}  {desc}".rstrip())
    # fallback 10 を最後に追加 (どの class にも map されない)
    lines.append(f"  10  (unclassified)  {_EXIT_CODE_DESCRIPTIONS[10]}")
    return "\n".join(lines) + "\n"


__all__ = [
    "LorebookError",
    "ConfigError",
    "LLMBackendUnavailableError",
    "AnalyzerInitError",
    "ZeroChunksError",
    "WikiGenerationError",
    "InputError",
    "EncodingError",
    "ChunkingError",
    "TfidfError",
    "EntityAggregationError",
    "AtomicSwapError",
    "RunReportError",
    "EXIT_CODE_MAP",
    "describe_exit_codes",
]
