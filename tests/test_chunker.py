"""Chunker の単体テスト. Ginza 非依存 (simple_japanese_splitter を使う)."""
from __future__ import annotations

import pytest

from chunking.chunker import (
    Chunker,
    simple_japanese_splitter,
)
from chunking.normalize import normalize_text


def _build_chunker(**kwargs) -> Chunker:
    defaults = dict(target_chars=50, overlap_chars=10, max_chunk_chars=150)
    defaults.update(kwargs)
    return Chunker(simple_japanese_splitter, **defaults)


def test_empty_text_yields_nothing() -> None:
    chunker = _build_chunker()
    assert list(chunker.chunk_document("a.txt", "")) == []


def test_single_short_sentence_makes_one_chunk() -> None:
    chunker = _build_chunker()
    text = "これは短い文です。"
    chunks = list(chunker.chunk_document("a.txt", text))
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(text)


def test_chunk_boundaries_respect_sentence_boundaries() -> None:
    chunker = _build_chunker(target_chars=30, overlap_chars=0, max_chunk_chars=80)
    text = (
        "あいうえおかきくけこ。"
        "さしすせそたちつてと。"
        "なにぬねのはひふへほ。"
        "まみむめもやゆよらり。"
    )
    chunks = list(chunker.chunk_document("a.txt", text))
    # ターゲット 30、各文 11字 → 文 3 本で 33 字を超える → 3 文/チャンク
    assert len(chunks) >= 2
    for c in chunks:
        # 各チャンクの末尾は「。」で終わる (文境界)
        assert c.text.endswith("。"), f"chunk {c.chunk_id!r} text {c.text!r} does not end on sentence boundary"


def test_overlap_pulls_previous_tail() -> None:
    chunker = _build_chunker(target_chars=30, overlap_chars=10, max_chunk_chars=80)
    text = (
        "あいうえおかきくけこ。"  # 11
        "さしすせそたちつてと。"  # 11
        "なにぬねのはひふへほ。"  # 11
        "まみむめもやゆよらり。"  # 11
    )
    chunks = list(chunker.chunk_document("a.txt", text))
    assert len(chunks) >= 2
    # 2 番目以降のチャンクは 1 番目チャンクの末尾とオーバーラップ開始位置を共有する
    second = chunks[1]
    first_end = chunks[0].char_end
    assert second.char_start < first_end, (
        f"second chunk char_start {second.char_start} should be < first chunk end {first_end}"
    )


def test_sentence_over_target_emits_standalone_chunk() -> None:
    chunker = _build_chunker(target_chars=20, overlap_chars=0, max_chunk_chars=200)
    long_sent = "あ" * 80 + "。"  # 81 chars, > target 20 だが < max 200
    text = f"最初の文です。{long_sent}次の文です。"
    chunks = list(chunker.chunk_document("a.txt", text))
    assert any(len(c.text) >= 80 for c in chunks)
    # warning が記録されている
    assert any(w.kind == "sentence_over_target" for w in chunker.warnings)


def test_sentence_over_max_triggers_soft_split() -> None:
    chunker = _build_chunker(target_chars=30, overlap_chars=0, max_chunk_chars=50)
    # 読点を含む長い 1 文を soft-split させる (100 chars > max 50)
    long_sent = "前半、" + ("あ" * 80) + "、後半です。"
    chunks = list(chunker.chunk_document("a.txt", long_sent))
    assert len(chunks) >= 2
    # warning が記録されている
    assert any(w.kind == "sentence_soft_split" for w in chunker.warnings)
    # 各チャンクが max + tolerance 以下
    for c in chunks:
        assert len(c.text) <= 60, f"chunk {len(c.text)} chars exceeds soft-split tolerance"


def test_chunk_id_is_stable_for_same_input() -> None:
    chunker1 = _build_chunker()
    chunker2 = _build_chunker()
    text = normalize_text("同じテキストを2回処理する。結果のchunk_idは一致するはず。")
    c1 = list(chunker1.chunk_document("a.txt", text))
    c2 = list(chunker2.chunk_document("a.txt", text))
    assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]


def test_chunk_id_changes_when_source_changes() -> None:
    chunker = _build_chunker()
    text = "同じ本文。別の source。"
    c1 = list(chunker.chunk_document("a.txt", text))
    c2 = list(chunker.chunk_document("b.txt", text))
    assert c1[0].chunk_id != c2[0].chunk_id


def test_chunk_id_changes_when_target_changes() -> None:
    """chunk_id は chunking パラメータに連動するため、target を変えると ID が入れ替わる."""
    chunker_small = _build_chunker(target_chars=20)
    chunker_large = _build_chunker(target_chars=100)
    text = (
        "あいうえおかきくけこ。"
        "さしすせそたちつてと。"
        "なにぬねのはひふへほ。"
    )
    c_small = list(chunker_small.chunk_document("a.txt", text))
    c_large = list(chunker_large.chunk_document("a.txt", text))
    # 少なくとも 1 つ以上 ID が異なる (正規化後テキストが同じでも offset が違う)
    assert {c.chunk_id for c in c_small} != {c.chunk_id for c in c_large}


def test_chunker_rejects_invalid_params() -> None:
    with pytest.raises(ValueError):
        Chunker(simple_japanese_splitter, target_chars=0)
    with pytest.raises(ValueError):
        Chunker(simple_japanese_splitter, overlap_chars=-1)
    with pytest.raises(ValueError):
        Chunker(simple_japanese_splitter, target_chars=500, max_chunk_chars=100)


def test_row_index_is_zero_based_and_sequential() -> None:
    chunker = _build_chunker(target_chars=30)
    text = (
        "あいうえおかきくけこ。"
        "さしすせそたちつてと。"
        "なにぬねのはひふへほ。"
        "まみむめもやゆよらり。"
    )
    chunks = list(chunker.chunk_document("a.txt", text))
    row_indices = [c.row_index for c in chunks]
    assert row_indices == list(range(len(chunks)))


def test_source_posix_relative_normalization() -> None:
    chunker = _build_chunker()
    text = "テスト。"
    chunks = list(chunker.chunk_document("samples/01_news.txt", text, input_dir="samples"))
    assert chunks[0].source == "01_news.txt"


def test_simple_splitter_handles_newlines() -> None:
    sents = list(simple_japanese_splitter("一行目。\n二行目。\n"))
    assert len(sents) == 2
    assert "一行目。" in sents[0]
