"""JapaneseAnalyzer のテスト.

Ginza + ja_ginza_electra 非インストール環境では skip する構造。
pip install -e . 後にフルスイートが走る。
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _ginza_available() -> bool:
    try:
        import spacy  # noqa: F401
    except ImportError:
        return False
    try:
        importlib.import_module("ja_ginza_electra")
        return True
    except ImportError:
        return False


requires_ginza = pytest.mark.skipif(
    not _ginza_available(),
    reason="ja_ginza_electra not installed; run `pip install -e .` to enable",
)


def test_schema_imports_without_spacy() -> None:
    from chunking.schema import AnalyzerConfig, EntityMention

    em = EntityMention(name="test", ner_label="PERSON", char_start=0, char_end=4)
    assert em.name == "test"
    cfg = AnalyzerConfig(strict_match={"a": 1}, compat_match={}, tfidf={})
    assert cfg.to_dict()["strict_match"] == {"a": 1}
    round_trip = AnalyzerConfig.from_dict(cfg.to_dict())
    assert round_trip.strict_match == cfg.strict_match


def test_analyzer_json_config_round_trip_without_spacy() -> None:
    """AnalyzerConfig の JSON 往復 (Ginza 不要)."""
    from chunking.schema import AnalyzerConfig

    original = AnalyzerConfig(
        strict_match={
            "model_name": "ja_ginza_electra",
            "split_mode": "C",
            "pos_allowlist": ["NOUN"],
            "stopwords": ["こと"],
            "normalization": {"nfkc": True, "lf_only": True, "strip_trailing": True, "collapse_spaces": True},
        },
        compat_match={"ginza": "5.2.0", "spacy": "3.7.4"},
        tfidf={"min_df": 1, "max_df": 0.95},
    )
    d = original.to_dict()
    restored = AnalyzerConfig.from_dict(d)
    assert restored.strict_match == original.strict_match
    assert restored.compat_match == original.compat_match
    assert restored.tfidf == original.tfidf


@requires_ginza
def test_iter_sentences_basic() -> None:
    from chunking.analyzer import JapaneseAnalyzer

    analyzer = JapaneseAnalyzer()
    sents = list(analyzer.iter_sentences("これは一文目です。これは二文目です。"))
    assert len(sents) == 2


@requires_ginza
def test_iter_sentences_empty() -> None:
    from chunking.analyzer import JapaneseAnalyzer

    analyzer = JapaneseAnalyzer()
    assert list(analyzer.iter_sentences("")) == []


@requires_ginza
def test_iter_entities_detects_person() -> None:
    from chunking.analyzer import JapaneseAnalyzer

    analyzer = JapaneseAnalyzer()
    ents = list(analyzer.iter_entities("田中太郎は東京都に住んでいる。"))
    labels = {e.ner_label for e in ents}
    # OntoNotes5 labels: PERSON / GPE (for 東京都) / LOC など
    assert any(label in labels for label in ("PERSON", "GPE", "LOC"))
    # span の char offset が意味のある範囲
    for e in ents:
        assert 0 <= e.char_start < e.char_end


@requires_ginza
def test_tokenize_for_tfidf_filters_particles() -> None:
    from chunking.analyzer import JapaneseAnalyzer

    analyzer = JapaneseAnalyzer()
    tokens = analyzer.tokenize_for_tfidf("東京に行きました。")
    # 助詞「に」「ました」は除外、「東京」「行く」は残る想定
    assert "東京" in tokens
    assert "に" not in tokens


@requires_ginza
def test_analyzer_json_round_trip(tmp_path: Path) -> None:
    from chunking.analyzer import JapaneseAnalyzer

    analyzer = JapaneseAnalyzer()
    path = tmp_path / "analyzer.json"
    analyzer.save(path)
    assert path.exists()
    loaded = JapaneseAnalyzer.load_and_verify(path)
    # 同じ設定で保存 → 復元 → 例外なし
    assert loaded.tokenize_for_tfidf("東京") == analyzer.tokenize_for_tfidf("東京")


@requires_ginza
def test_analyzer_json_strict_mismatch(tmp_path: Path) -> None:
    import json
    from chunking.analyzer import JapaneseAnalyzer
    from chunking.schema import AnalyzerVersionMismatchError

    analyzer = JapaneseAnalyzer()
    path = tmp_path / "analyzer.json"
    analyzer.save(path)
    # split_mode を書き換えて不整合を作る
    data = json.loads(path.read_text("utf-8"))
    data["strict_match"]["split_mode"] = "A"
    path.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    with pytest.raises(AnalyzerVersionMismatchError):
        JapaneseAnalyzer.load_and_verify(path)
