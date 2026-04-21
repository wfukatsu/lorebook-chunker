"""normalize_text の単体テスト. 外部依存なし."""
from __future__ import annotations

from chunking.normalize import NORMALIZATION_SPEC, normalize_text


def test_nfkc_fullwidth_to_halfwidth() -> None:
    assert normalize_text("Ｃａｆｅ") == "Cafe"
    assert normalize_text("１２３") == "123"


def test_crlf_and_cr_to_lf() -> None:
    assert normalize_text("a\r\nb\r\nc") == "a\nb\nc"
    assert normalize_text("a\rb") == "a\nb"


def test_trailing_whitespace_per_line() -> None:
    # 末尾の空白 (trailing `\n` を含む) は strip 段階で除去される
    assert normalize_text("abc   \ndef\t\n") == "abc\ndef"


def test_collapse_consecutive_spaces() -> None:
    assert normalize_text("a  b   c") == "a b c"


def test_ideographic_space_is_collapsed() -> None:
    # IDEOGRAPHIC SPACE U+3000 → NFKC で half-width space になり、折りたたまれる
    assert normalize_text("a　b") == "a b"
    assert normalize_text("a　 　b") == "a b"


def test_idempotent() -> None:
    samples = [
        "Ｃａｆｅ　\r\n  \r\n テスト ",
        "普通の文章。",
        "",
        "a\tb\tc",
        "改行だけ\n\n\n",
    ]
    for s in samples:
        once = normalize_text(s)
        twice = normalize_text(once)
        assert once == twice, f"not idempotent: {s!r} -> {once!r} -> {twice!r}"


def test_empty_string() -> None:
    assert normalize_text("") == ""


def test_bom_stripped_by_nfkc() -> None:
    # NFKC は BOM を残すが、input 時点で strip される想定なので documentation test
    # Python の open(..., encoding="utf-8-sig") で BOM は消える。ここでは normalize が
    # 壊さないことのみ確認
    assert "﻿" in normalize_text("﻿test")


def test_normalization_spec_exported() -> None:
    assert NORMALIZATION_SPEC == {
        "nfkc": True,
        "lf_only": True,
        "strip_trailing": True,
        "collapse_spaces": True,
        "trim": True,
    }


def test_mixed_real_world() -> None:
    s = "  Ｈｅｌｌｏ　　世界  \r\n\tテスト　です  \r\n"
    expected = "Hello 世界\n テスト です"
    assert normalize_text(s) == expected
