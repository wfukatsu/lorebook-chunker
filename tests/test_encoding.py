"""U3: encoding detection pipeline のテスト.

- utf-8-sig strict 先行 (BOM 有無を透過吸収)
- `--encoding <name>` 明示時は detector を呼ばず strict decode
- `auto` + charset-normalizer 導入時: Shift-JIS / CP932 / EUC-JP を検出
- `auto` + charset-normalizer 未導入: utf8_failed_detector_unavailable エラー
- 短ファイル (< 100 bytes): too_short_to_detect
- chaos 高 / no language hit: detection_ambiguous

charset-normalizer は `[full]` extra (soft dep). import 可能性を
`lorebook_chunker.encoding._CHARSET_NORMALIZER_AVAILABLE` で検査する.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lorebook_chunker import encoding as enc_mod
from lorebook_chunker.encoding import (
    DEFAULT_CHAOS_THRESHOLD,
    MIN_SIZE_FOR_DETECTION,
    detect_encoding,
)
from lorebook_chunker.errors import ConfigError, EncodingError

FIXTURES = Path(__file__).parent / "fixtures" / "encoding_samples"


# ---- Happy path: utf-8 (BOM 有無統一) ------------------------------------


def test_plain_utf8_returns_utf8_sig() -> None:
    """plain UTF-8 (no BOM) は utf-8-sig codec で通る.

    codec 名は "utf-8-sig" を返す (utf-8 でも utf-8-sig でも decode 挙動は同じ).
    """
    text, actual = detect_encoding(FIXTURES / "plain_utf8.txt")
    assert actual == "utf-8-sig"
    assert "田中" in text
    # BOM 文字 (U+FEFF) が本文に残らない
    assert not text.startswith("﻿")


def test_utf8_bom_strips_bom() -> None:
    """UTF-8 BOM 付きファイルも utf-8-sig で通り、本文に BOM が残らない."""
    text, actual = detect_encoding(FIXTURES / "utf8_bom.txt")
    assert actual == "utf-8-sig"
    assert not text.startswith("﻿")
    assert text.startswith("田中")


def test_explicit_utf8_override_does_not_invoke_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`override != "auto"` の場合 charset_normalizer.from_path は呼ばれない."""
    called = {"n": 0}

    # charset_normalizer 未導入環境でも mock できるように、モジュール参照側を差し替え
    class _Sentinel:
        @staticmethod
        def from_path(*args: object, **kwargs: object) -> object:  # pragma: no cover
            called["n"] += 1
            raise AssertionError("detector should not be called when override is set")

    monkeypatch.setattr(enc_mod, "_charset_normalizer", _Sentinel)
    monkeypatch.setattr(enc_mod, "_CHARSET_NORMALIZER_AVAILABLE", True)

    text, actual = detect_encoding(FIXTURES / "plain_utf8.txt", override="utf-8")
    assert actual == "utf-8"
    assert "田中" in text
    assert called["n"] == 0


# ---- Detector-required happy paths -----------------------------------


@pytest.mark.skipif(
    not enc_mod._CHARSET_NORMALIZER_AVAILABLE,
    reason="charset-normalizer not installed (pip install -e '.[full]')",
)
def test_shift_jis_auto_detect() -> None:
    text, actual = detect_encoding(FIXTURES / "shift_jis.txt")
    # charset-normalizer は "shift_jis" or "cp932" (superset) を返す. 実装では
    # CP932 にまとめる傾向があるため、両者を許容する.
    assert actual in ("shift_jis", "sjis", "cp932", "shift-jis")
    assert "田中" in text
    assert "スカラー商事" in text


@pytest.mark.skipif(
    not enc_mod._CHARSET_NORMALIZER_AVAILABLE,
    reason="charset-normalizer not installed (pip install -e '.[full]')",
)
def test_cp932_halfwidth_katakana_auto_detect() -> None:
    text, actual = detect_encoding(FIXTURES / "cp932.txt")
    assert actual in ("cp932", "shift_jis", "sjis", "shift-jis")
    assert "ｱｲｳｴｵ" in text or "ｱｲｳ" in text
    assert "スカラー商事" in text


@pytest.mark.skipif(
    not enc_mod._CHARSET_NORMALIZER_AVAILABLE,
    reason="charset-normalizer not installed (pip install -e '.[full]')",
)
def test_euc_jp_auto_detect() -> None:
    text, actual = detect_encoding(FIXTURES / "euc_jp.txt")
    # 検出結果は euc_jp / euc-jp / eucjp / cp51932 のいずれかが典型.
    # 精密一致ではなく "euc" substring で判定する (libraries 間の正規化差を吸収).
    assert "euc" in actual.lower() or actual in ("cp51932",)
    assert "田中" in text


# ---- soft-import 失敗パス (charset-normalizer 非導入 emulation) --------


def test_utf8_sig_ok_when_detector_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """charset-normalizer 未導入でも utf-8-sig が通れば detect_encoding は成功."""
    monkeypatch.setattr(enc_mod, "_CHARSET_NORMALIZER_AVAILABLE", False)
    monkeypatch.setattr(enc_mod, "_charset_normalizer", None)
    text, actual = detect_encoding(FIXTURES / "plain_utf8.txt")
    assert actual == "utf-8-sig"
    assert "田中" in text


def test_non_utf8_without_detector_raises_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """charset-normalizer 未導入かつ utf-8-sig 不通 → EncodingError に install hint."""
    monkeypatch.setattr(enc_mod, "_CHARSET_NORMALIZER_AVAILABLE", False)
    monkeypatch.setattr(enc_mod, "_charset_normalizer", None)
    # >= MIN_SIZE_FOR_DETECTION bytes の非 UTF-8 を用意 (short path を避けるため)
    sample = (
        "田中はスカラー商事で働いています。東京に住んでいます。"
        "佐藤もスカラー商事の社員であり、大阪に住んでいます。"
    )
    target = tmp_path / "sjis.txt"
    target.write_bytes(sample.encode("shift_jis"))
    assert target.stat().st_size >= MIN_SIZE_FOR_DETECTION

    with pytest.raises(EncodingError) as excinfo:
        detect_encoding(target)
    err = excinfo.value
    assert err.context["reason"] == "utf8_failed_detector_unavailable"
    assert "install charset-normalizer" in err.context["hint"]
    assert err.exit_code == 12
    assert isinstance(err.__cause__, UnicodeDecodeError)


# ---- 短ファイル / あいまい検出 -----------------------------------------


@pytest.mark.skipif(
    not enc_mod._CHARSET_NORMALIZER_AVAILABLE,
    reason="too_short_to_detect branch only fires when detector is available",
)
def test_short_non_utf8_raises_too_short(tmp_path: Path) -> None:
    """< 100 bytes の非 UTF-8 は検出不能として EncodingError(too_short_to_detect)."""
    target = tmp_path / "short.txt"
    target.write_bytes("ｱｲｳ".encode("shift_jis"))
    assert target.stat().st_size < MIN_SIZE_FOR_DETECTION

    with pytest.raises(EncodingError) as excinfo:
        detect_encoding(target)
    assert excinfo.value.context["reason"] == "too_short_to_detect"
    assert excinfo.value.context["size_bytes"] < MIN_SIZE_FOR_DETECTION


@pytest.mark.skipif(
    not enc_mod._CHARSET_NORMALIZER_AVAILABLE,
    reason="charset-normalizer not installed (pip install -e '.[full]')",
)
def test_detection_ambiguous_when_chaos_above_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """chaos >= threshold の候補は reject されて detection_ambiguous エラー."""
    target = tmp_path / "ambig.txt"
    target.write_bytes(b"\x81\x82" * 80)  # ~160 bytes, above MIN_SIZE
    assert target.stat().st_size >= MIN_SIZE_FOR_DETECTION

    class _FakeMatch:
        encoding = "shift_jis"
        chaos = 0.5
        coherence = 0.1
        language = "Japanese"
        languages = ["Japanese"]

        def __str__(self) -> str:
            return "decoded-text"

    class _FakeMatches:
        _items = [_FakeMatch()]

        def best(self) -> _FakeMatch:
            return self._items[0]

        def __iter__(self):
            return iter(self._items)

    class _FakeCharsetNormalizer:
        @staticmethod
        def from_path(*args: object, **kwargs: object) -> _FakeMatches:
            return _FakeMatches()

    monkeypatch.setattr(enc_mod, "_charset_normalizer", _FakeCharsetNormalizer)
    monkeypatch.setattr(enc_mod, "_CHARSET_NORMALIZER_AVAILABLE", True)

    with pytest.raises(EncodingError) as excinfo:
        detect_encoding(target, chaos_threshold=DEFAULT_CHAOS_THRESHOLD)
    err = excinfo.value
    assert err.context["reason"] == "detection_ambiguous"
    cands = err.context["detected_candidates"]
    assert isinstance(cands, list) and len(cands) >= 1
    assert cands[0]["encoding"] == "shift_jis"
    assert cands[0]["chaos"] == pytest.approx(0.5)


@pytest.mark.skipif(
    not enc_mod._CHARSET_NORMALIZER_AVAILABLE,
    reason="charset-normalizer not installed (pip install -e '.[full]')",
)
def test_detection_ambiguous_when_no_language_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """languages == [] の候補は reject (ASCII-only ノイズ判定)."""
    target = tmp_path / "ambig2.txt"
    target.write_bytes(b"hello world ascii content, but long enough" * 4)

    class _FakeMatch:
        encoding = "ascii"
        chaos = 0.0
        coherence = 0.0
        language = "Unknown"
        languages: list[str] = []

        def __str__(self) -> str:
            return "decoded"

    class _FakeMatches:
        _items = [_FakeMatch()]

        def best(self) -> _FakeMatch:
            return self._items[0]

        def __iter__(self):
            return iter(self._items)

    class _FakeCharsetNormalizer:
        @staticmethod
        def from_path(*args: object, **kwargs: object) -> _FakeMatches:
            return _FakeMatches()

    monkeypatch.setattr(enc_mod, "_charset_normalizer", _FakeCharsetNormalizer)
    monkeypatch.setattr(enc_mod, "_CHARSET_NORMALIZER_AVAILABLE", True)

    # このテストは utf-8-sig が通らない bytes を用意しておく必要がある.
    target.write_bytes(b"\x81\x82" * 80)
    with pytest.raises(EncodingError) as excinfo:
        detect_encoding(target)
    assert excinfo.value.context["reason"] == "detection_ambiguous"


# ---- エラーパス: strict decode ---------------------------------------


def test_explicit_utf8_on_invalid_bytes_raises_encoding_error(tmp_path: Path) -> None:
    """utf-8 明示で非 UTF-8 bytes を与えると EncodingError(byte_offset=int)."""
    target = tmp_path / "bad.txt"
    target.write_bytes(b"\xff\xfe\x00 invalid")

    with pytest.raises(EncodingError) as excinfo:
        detect_encoding(target, override="utf-8")
    err = excinfo.value
    assert err.exit_code == 12
    assert err.context["reason"] == "explicit_decode_failed"
    assert err.context["encoding_attempted"] == "utf-8"
    assert isinstance(err.context["byte_offset"], int)
    assert isinstance(err.__cause__, UnicodeDecodeError)


def test_invalid_codec_name_raises_config_error_via_cli_validation() -> None:
    """無効 codec 名は ConfigError(reason="invalid_encoding"). 検証は ingest
    層 (_validate_encoding_option) で起きる — detect_encoding 内では行わない.
    """
    from lorebook_chunker.ingest import _validate_encoding_option

    with pytest.raises(ConfigError) as excinfo:
        _validate_encoding_option("foo-encoding-that-does-not-exist")
    assert excinfo.value.exit_code == 2
    assert excinfo.value.context["reason"] == "invalid_encoding"
    assert excinfo.value.context["encoding"] == "foo-encoding-that-does-not-exist"


def test_valid_encoding_names_accepted() -> None:
    """"auto" と代表的な codec 名は _validate_encoding_option で通る."""
    from lorebook_chunker.ingest import _validate_encoding_option

    for name in ["auto", "utf-8", "utf-8-sig", "cp932", "shift_jis", "euc_jp"]:
        _validate_encoding_option(name)  # no raise


# ---- Integration: IngestRunner 経由で Shift-JIS 日本語が chunks に入る ---


@pytest.mark.skipif(
    not enc_mod._CHARSET_NORMALIZER_AVAILABLE,
    reason="charset-normalizer not installed (pip install -e '.[full]')",
)
def test_ingest_shift_jis_with_auto_encoding_produces_decoded_chunks(
    tmp_path: Path,
) -> None:
    """end-to-end: Shift-JIS ファイル + `encoding="auto"` で chunks.jsonl に
    正しく decode された日本語が載る (mojibake しない)."""
    import json

    from lorebook_chunker.ingest import IngestConfig, IngestRunner
    from tests.conftest import _ScriptedLLM, _StubAnalyzer

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    sample = (
        "田中はスカラー商事で働いています。東京に住んでいます。"
        "佐藤もスカラー商事の社員で、大阪に住んでいます。"
        "プロジェクトは順調に進行中です。"
    )
    (input_dir / "01.txt").write_bytes(sample.encode("shift_jis"))
    output_dir = tmp_path / "out"

    cfg = IngestConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        skip_wiki=True,
        target_chars=30,
        max_chunk_chars=200,
        encoding="auto",
        show_progress=False,
    )
    runner = IngestRunner(
        cfg,
        analyzer_factory=lambda path: _StubAnalyzer(),
        llm_factory=lambda backend, config: _ScriptedLLM([]),
    )
    result = runner.run()
    assert result.exit_code == 0, result.errors
    assert result.skipped_files == 0
    chunks_path = output_dir / "chunks.jsonl"
    assert chunks_path.exists()
    found = False
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "田中" in rec["text"] and "スカラー商事" in rec["text"]:
                found = True
                break
    assert found, "Shift-JIS decoded Japanese should appear in chunks.jsonl"
