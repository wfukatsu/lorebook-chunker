"""テキスト正規化ロジック (R18 source_hash + R7 analyzer 共通前処理).

契約 (idempotent):
1. NFKC 正規化
2. 改行を LF に統一 (CRLF / CR → LF)
3. 各行の末尾空白を除去
4. 連続する空白 (半角スペース・タブ・IDEOGRAPHIC SPACE U+3000) を 1 つに折りたたむ
5. 全体の先頭/末尾空白 (strip) を除去

この定義は analyzer.json にも記録し、R18 の source_hash 計算でも同一定義を使う。
"""
from __future__ import annotations

import re
import unicodedata

_COLLAPSIBLE_WS = re.compile(r"[ \t　]+")
_TRAILING_WS_PER_LINE = re.compile(r"[ \t　]+$", re.MULTILINE)


def normalize_text(s: str) -> str:
    """正規化: NFKC + LF 統一 + 末尾空白除去 + 連続空白折りたたみ + 全体 strip.

    Idempotent: ``normalize_text(normalize_text(x)) == normalize_text(x)``.
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _TRAILING_WS_PER_LINE.sub("", s)
    s = _COLLAPSIBLE_WS.sub(" ", s)
    s = s.strip()
    return s


NORMALIZATION_SPEC: dict[str, bool] = {
    "nfkc": True,
    "lf_only": True,
    "strip_trailing": True,
    "collapse_spaces": True,
    "trim": True,
}
