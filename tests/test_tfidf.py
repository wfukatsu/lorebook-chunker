"""TF-IDF builder の単体テスト. Ginza 非依存 (whitespace-splitting analyzer を注入)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from chunking.tfidf import TfidfBuilder


def ws_analyzer(text: str) -> list[str]:
    """テスト用: 空白で分割する素朴な analyzer."""
    return [tok for tok in text.split() if tok]


@pytest.fixture
def builder() -> TfidfBuilder:
    return TfidfBuilder(ws_analyzer, top_keywords=3, min_df=1, max_df=1.0)


@pytest.fixture
def corpus() -> list[str]:
    return [
        "alpha beta gamma",
        "beta gamma delta",
        "gamma delta epsilon",
        "alpha epsilon zeta",
    ]


def test_fit_transform_returns_valid_shapes(builder: TfidfBuilder, corpus: list[str]) -> None:
    matrix, vocab, idf = builder.fit_transform(corpus)
    assert matrix.shape[0] == len(corpus)
    assert matrix.shape[1] == len(vocab)
    assert idf.shape == (len(vocab),)
    assert all(isinstance(term, str) for term in vocab.keys())


def test_top_keywords_per_chunk_are_sorted(builder: TfidfBuilder, corpus: list[str]) -> None:
    matrix, vocab, _ = builder.fit_transform(corpus)
    top = builder.top_keywords_per_chunk(matrix, vocab)
    assert len(top) == len(corpus)
    for row in top:
        assert len(row) <= 3  # top_keywords=3
        values = [kw["tfidf"] for kw in row]
        assert values == sorted(values, reverse=True)
        assert all(isinstance(kw["term"], str) for kw in row)


def test_save_load_round_trip(builder: TfidfBuilder, corpus: list[str], tmp_path: Path) -> None:
    matrix, vocab, idf = builder.fit_transform(corpus)
    path = tmp_path / "vocab.npz"
    builder.save(path, matrix, vocab, idf)
    assert path.exists()
    arts = TfidfBuilder.load(path)
    assert arts.matrix.shape == matrix.shape
    np.testing.assert_allclose(arts.matrix.toarray(), matrix.toarray())
    assert arts.vocabulary == vocab
    np.testing.assert_allclose(arts.idf, idf)
    assert arts.config["min_df"] == 1
    assert arts.config["top_keywords"] == 3


def test_rebuild_vectorizer_and_transform_query(
    builder: TfidfBuilder, corpus: list[str]
) -> None:
    matrix, vocab, idf = builder.fit_transform(corpus)
    config = builder._build_config()
    rebuilt = builder.rebuild_vectorizer(ws_analyzer, vocab, idf, config)
    vec = builder.transform_query(rebuilt, "alpha beta")
    assert isinstance(vec, csr_matrix)
    assert vec.shape == (1, len(vocab))
    # "alpha" と "beta" の index が non-zero
    alpha_idx = vocab["alpha"]
    beta_idx = vocab["beta"]
    dense = vec.toarray().flatten()
    assert dense[alpha_idx] > 0
    assert dense[beta_idx] > 0
    # 非含有トークンは 0
    gamma_delta_epsilon = [vocab[t] for t in ("gamma", "delta", "epsilon", "zeta")]
    for idx in gamma_delta_epsilon:
        assert dense[idx] == 0


def test_empty_corpus_raises(builder: TfidfBuilder) -> None:
    with pytest.raises(ValueError):
        builder.fit_transform([])


def test_single_chunk_corpus(builder: TfidfBuilder) -> None:
    matrix, vocab, idf = builder.fit_transform(["alone token bag"])
    assert matrix.shape[0] == 1
    assert matrix.shape[1] == 3  # alone, token, bag


def test_oov_query_returns_zero_vector(builder: TfidfBuilder, corpus: list[str]) -> None:
    matrix, vocab, idf = builder.fit_transform(corpus)
    config = builder._build_config()
    rebuilt = builder.rebuild_vectorizer(ws_analyzer, vocab, idf, config)
    vec = builder.transform_query(rebuilt, "完全に別の語 unknown_token")
    # 全ての語彙と一致しないので全 0
    assert vec.nnz == 0
