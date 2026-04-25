"""Staged encoding detection: explicit override > `utf-8-sig` strict > charset-normalizer.

U3 で導入. `read_text(encoding="utf-8")` 一本だった旧 ingest.py を置き換え、

1. `override != "auto"` → `path.read_text(encoding=override)` strict のみ試行
2. `override == "auto"` → `utf-8-sig` strict 先行 (BOM 有無を透過的に吸収)
3. それでも失敗し、かつ `charset-normalizer` が import 可能 → 検出を試みる
4. 検出不能 / あいまいなら `EncodingError` で明示的な `--encoding` 指定を要求

`charset-normalizer` は **soft dependency** — `[full]` extra でのみ導入される.
未導入環境では UTF-8 (BOM 有無問わず) のみ通過し、非 UTF-8 入力は
`EncodingError(reason="utf8_failed_detector_unavailable")` で skip される.

Threshold rationale:
- `chaos < 0.3`: 公式 README 推奨. 0.1 以下は当該 encoding で完全 decode 可能,
  0.3 前後は「可能性が高い」水準. 0.5 近辺は曖昧 → reject.
- "language hit" を要件に加えるのは ASCII-only のノイズを除外するため
  (charset-normalizer は ASCII decodable なバイト列すべてに `encoding=ascii`
  `chaos=0.0` を返しうる). ただし ASCII-only 入力は utf-8-sig 側で通るので、
  ここに来るのは非 UTF-8 = 言語推定が効くケースがほぼ全て.
- `size_bytes < 100`: 短文は統計的検出が不安定なので、plain UTF-8 に失敗した
  時点で `--encoding` 明示を要求する.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from lorebook_chunker.errors import EncodingError

# charset-normalizer は `[full]` extra 経由で導入される soft dependency.
# 未導入でも本モジュールは import でき、utf-8-sig path は機能する.
try:  # pragma: no cover — import availability は module load 時に決まる
    import charset_normalizer as _charset_normalizer  # type: ignore

    _CHARSET_NORMALIZER_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _charset_normalizer = None  # type: ignore[assignment]
    _CHARSET_NORMALIZER_AVAILABLE = False


# 公開定数 (test が直接参照する)
DEFAULT_CHAOS_THRESHOLD: float = 0.3
MIN_SIZE_FOR_DETECTION: int = 100


def detect_encoding_bytes(
    data: bytes,
    *,
    override: str = "auto",
    chaos_threshold: float = DEFAULT_CHAOS_THRESHOLD,
) -> tuple[str, str]:
    """byte 列に対する軽量 encoding probe. ``detect_encoding`` の path 版と同じ
    staged pipeline を data: bytes に対して実行する.

    U7 の ``ingest --dry-run`` が最初のファイルの先頭 64 KiB sample に対して
    書き込みなしで encoding を推定するために使う. path 版と違って `EncodingError`
    は raise しない (dry-run は pre-flight であり、skip/fail を早い段階で
    宣言する情報価値が低いため). 代わりに best-effort で
    ``(decoded_or_empty, encoding_or_"unknown")`` を返す.

    Args:
        data: 対象 bytes (ファイル先頭のサンプルで十分)
        override: ``"auto"`` (既定) または明示 codec 名. 非 auto で decode 失敗
            時は ``("", "decode_failed")`` を返す.

    Returns:
        ``(decoded_text, encoding_name)``. 推定不能時は ``("", "unknown")``.
    """
    if override != "auto":
        try:
            return data.decode(override), override
        except (UnicodeDecodeError, LookupError):
            return "", "decode_failed"

    # auto: utf-8-sig を先に試す.
    try:
        return data.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        pass

    if not _CHARSET_NORMALIZER_AVAILABLE:
        return "", "unknown"

    # charset-normalizer を試す.
    assert _charset_normalizer is not None
    try:
        matches = _charset_normalizer.from_bytes(data, steps=5, chunk_size=512)
        best = matches.best()
    except Exception:  # pragma: no cover - library-side failures
        return "", "unknown"
    if best is None:
        return "", "unknown"
    chaos = float(getattr(best, "chaos", 1.0))
    if chaos >= chaos_threshold:
        return "", "unknown"
    return str(best), str(best.encoding)


def detect_encoding(
    path: Path,
    *,
    override: str = "auto",
    chaos_threshold: float = DEFAULT_CHAOS_THRESHOLD,
) -> tuple[str, str]:
    """ファイルを decode し、`(text, actual_encoding)` を返す.

    Args:
        path: 対象ファイル.
        override: `"auto"` (既定) なら staged detection. それ以外は strict 明示.
        chaos_threshold: `charset-normalizer` の chaos score 上限. default 0.3.

    Raises:
        EncodingError: 明示指定の decode 失敗 / auto 検出失敗 / 検出器未導入 /
            短すぎて検出不能 / 検出結果があいまい.
    """
    if override != "auto":
        return _decode_strict(path, override)

    # auto: utf-8-sig を先に試す. BOM 付き / 無しどちらも通る.
    try:
        text = path.read_text(encoding="utf-8-sig")
        return text, "utf-8-sig"
    except UnicodeDecodeError as utf8_exc:
        # 次段の判断のために byte 列を取得 (Path.stat でサイズだけ見てもよいが、
        # EncodingError context 用に size_bytes を拾う).
        data = path.read_bytes()
        size_bytes = len(data)

        if not _CHARSET_NORMALIZER_AVAILABLE:
            raise EncodingError(
                f"utf-8 decode failed and encoding detector unavailable: {path}",
                reason="utf8_failed_detector_unavailable",
                path=str(path),
                size_bytes=size_bytes,
                byte_offset=getattr(utf8_exc, "start", None),
                hint=(
                    "install charset-normalizer via [full] extra "
                    "(`pip install -e '.[full]'`), or pass --encoding explicitly"
                ),
            ) from utf8_exc

        # 検出不能 (短すぎ). plain UTF-8 も通らなかった時点で明示指定を要求する.
        if size_bytes < MIN_SIZE_FOR_DETECTION:
            raise EncodingError(
                f"file too short for encoding detection: {path} ({size_bytes} bytes)",
                reason="too_short_to_detect",
                path=str(path),
                size_bytes=size_bytes,
                byte_offset=getattr(utf8_exc, "start", None),
                hint=(
                    f"auto-detection needs at least {MIN_SIZE_FOR_DETECTION} bytes; "
                    "pass --encoding explicitly"
                ),
            ) from utf8_exc

        return _detect_with_charset_normalizer(
            path, data, chaos_threshold, utf8_exc
        )


def _decode_strict(path: Path, encoding: str) -> tuple[str, str]:
    """明示 encoding で strict decode. 失敗時は EncodingError に wrap."""
    try:
        text = path.read_text(encoding=encoding)
        return text, encoding
    except UnicodeDecodeError as exc:
        size_bytes: int | None
        try:
            size_bytes = path.stat().st_size
        except OSError:  # pragma: no cover
            size_bytes = None
        raise EncodingError(
            f"utf-8 decode failed, explicit encoding={encoding!r} rejected {path}",
            reason="explicit_decode_failed",
            path=str(path),
            encoding_attempted=encoding,
            size_bytes=size_bytes,
            byte_offset=getattr(exc, "start", None),
        ) from exc


def _detect_with_charset_normalizer(
    path: Path,
    data: bytes,
    chaos_threshold: float,
    utf8_exc: UnicodeDecodeError,
) -> tuple[str, str]:
    """charset-normalizer で auto 検出. 失敗時は EncodingError.

    `data` / `utf8_exc` は呼び出し元で取得済みのものを再利用する (I/O 節約).
    """
    assert _charset_normalizer is not None  # _CHARSET_NORMALIZER_AVAILABLE=True の前提
    matches = _charset_normalizer.from_path(path, steps=5, chunk_size=512)
    best = matches.best()

    candidates = _format_candidates(matches)

    if best is None:
        raise EncodingError(
            f"encoding detection found no viable candidate: {path}",
            reason="detection_ambiguous",
            path=str(path),
            size_bytes=len(data),
            byte_offset=getattr(utf8_exc, "start", None),
            detected_candidates=candidates,
            hint="no encoding candidate passed; pass --encoding explicitly",
        ) from utf8_exc

    chaos = float(getattr(best, "chaos", 1.0))
    # charset-normalizer は scalar `.language` (top guess, e.g. "Japanese") と
    # list `.languages` (multi-agreement) を両方持つ. 単一ファイル検出では
    # `.languages` は空になりがち (複数 alphabet 一致が要件) なので、primary
    # signal には scalar `.language` を使う. "Unknown" は言語不詳とみなす.
    language: str = str(getattr(best, "language", "") or "")
    languages_list = list(getattr(best, "languages", []) or [])
    has_language_hit = (
        bool(languages_list)
        or (language and language.lower() not in ("", "unknown"))
    )

    # plan Approach: chaos < threshold かつ language hit >= 1.
    if chaos >= chaos_threshold or not has_language_hit:
        raise EncodingError(
            f"encoding detection ambiguous: {path} "
            f"(best={best.encoding} chaos={chaos:.3f} language={language!r})",
            reason="detection_ambiguous",
            path=str(path),
            size_bytes=len(data),
            byte_offset=getattr(utf8_exc, "start", None),
            detected_candidates=candidates,
            hint="pass --encoding explicitly",
        ) from utf8_exc

    encoding = str(best.encoding)
    # charset-normalizer は bytes を既に decode 済みの `best` を持っているので str 化して返す.
    text = str(best)
    return text, encoding


def _format_candidates(matches: Any) -> list[dict[str, Any]]:
    """`CharsetMatches` を top 候補 3 件まで JSON 可搬な dict list に整形."""
    out: list[dict[str, Any]] = []
    # matches は iterable + indexable. top 3 までに絞る (log.md / run_report 可読性).
    try:
        iterator = list(matches)[:3]
    except TypeError:  # pragma: no cover
        iterator = []
    for m in iterator:
        out.append(
            {
                "encoding": str(getattr(m, "encoding", "")),
                "chaos": float(getattr(m, "chaos", 1.0)),
                "coherence": float(getattr(m, "coherence", 0.0)),
                # scalar `language` は top guess (e.g. "Japanese" / "Unknown").
                "language": str(getattr(m, "language", "") or ""),
                # 複数一致 list も残すが単独 file では空になりがち.
                "languages": list(getattr(m, "languages", []) or []),
            }
        )
    return out


__all__ = [
    "detect_encoding",
    "detect_encoding_bytes",
    "DEFAULT_CHAOS_THRESHOLD",
    "MIN_SIZE_FOR_DETECTION",
    "_CHARSET_NORMALIZER_AVAILABLE",
]
