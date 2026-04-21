"""TF-IDF builder: custom analyzer, single-file .npz persistence (matrix + vocab + idf + config).

`scipy.sparse.save_npz` 単独では疎行列しか保存できないため、`np.savez_compressed` で
matrix の CSR 配列・vocabulary・idf・config をまとめて単一 `.npz` に格納する。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# analyzer callable 契約: str -> list[str]
AnalyzerCallable = Callable[[str], list[str]]


@dataclass
class TfidfArtifacts:
    """vocab.npz の復元表現."""

    matrix: csr_matrix
    vocabulary: dict[str, int]
    idf: np.ndarray
    config: dict[str, Any] = field(default_factory=dict)


class TfidfBuilder:
    """TfidfVectorizer のラッパ.

    同じ analyzer で fit / transform を完結させる。fitted vectorizer は pickle しない
    (spaCy Language オブジェクトを含むため)。代わりに vocabulary / idf / config を保存して
    クエリ時に再構成する。
    """

    def __init__(
        self,
        analyzer: AnalyzerCallable,
        *,
        top_keywords: int = 10,
        min_df: int | float = 1,
        max_df: int | float = 0.95,
        sublinear_tf: bool = True,
    ) -> None:
        self._analyzer = analyzer
        self.top_keywords = top_keywords
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self._vectorizer: TfidfVectorizer | None = None

    # ---- fit / keywords ----------------------------------------------

    def fit_transform(self, texts: Sequence[str]) -> tuple[csr_matrix, dict[str, int], np.ndarray]:
        if not texts:
            raise ValueError("fit_transform: texts must be non-empty")
        self._vectorizer = TfidfVectorizer(
            analyzer=self._analyzer,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
        )
        matrix = self._vectorizer.fit_transform(list(texts))
        # fit_transform の返り値型: scipy.sparse.csr_matrix
        if not isinstance(matrix, csr_matrix):
            matrix = csr_matrix(matrix)
        vocabulary: dict[str, int] = dict(self._vectorizer.vocabulary_)
        idf = np.asarray(self._vectorizer.idf_, dtype=np.float64)
        return matrix, vocabulary, idf

    def top_keywords_per_chunk(
        self, matrix: csr_matrix, vocabulary: dict[str, int]
    ) -> list[list[dict[str, Any]]]:
        """各行の TF-IDF 上位 N 語を返す."""
        inv_vocab = {idx: term for term, idx in vocabulary.items()}
        rows: list[list[dict[str, Any]]] = []
        for row_idx in range(matrix.shape[0]):
            row = matrix.getrow(row_idx)
            if row.nnz == 0:
                rows.append([])
                continue
            # getrow は sparse。data と indices を取って降順ソート
            values = np.asarray(row.data)
            indices = np.asarray(row.indices)
            order = np.argsort(-values)[: self.top_keywords]
            rows.append(
                [
                    {"term": inv_vocab[int(indices[i])], "tfidf": float(values[i])}
                    for i in order
                ]
            )
        return rows

    # ---- persistence -------------------------------------------------

    def save(
        self,
        path: str | Path,
        matrix: csr_matrix,
        vocabulary: dict[str, int],
        idf: np.ndarray,
    ) -> None:
        """vocab.npz に matrix + vocabulary + idf + config を統合保存."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = self._build_config()
        # vocabulary は idx 順の terms 配列で保存 (dict は np.save できないため)
        terms_in_order = [""] * len(vocabulary)
        for term, idx in vocabulary.items():
            terms_in_order[idx] = term
        vocab_array = np.array(terms_in_order, dtype=object)
        np.savez_compressed(
            path,
            matrix_data=matrix.data,
            matrix_indices=matrix.indices,
            matrix_indptr=matrix.indptr,
            matrix_shape=np.asarray(matrix.shape, dtype=np.int64),
            vocabulary_terms=vocab_array,
            idf=idf,
            config_json=np.array(json.dumps(config), dtype=object),
        )

    @staticmethod
    def load(path: str | Path) -> TfidfArtifacts:
        path = Path(path)
        with np.load(path, allow_pickle=True) as data:
            matrix = csr_matrix(
                (
                    data["matrix_data"],
                    data["matrix_indices"],
                    data["matrix_indptr"],
                ),
                shape=tuple(data["matrix_shape"]),
            )
            terms = list(data["vocabulary_terms"])
            vocabulary = {str(term): idx for idx, term in enumerate(terms)}
            idf = np.asarray(data["idf"], dtype=np.float64)
            config = json.loads(str(data["config_json"]))
        return TfidfArtifacts(
            matrix=matrix,
            vocabulary=vocabulary,
            idf=idf,
            config=config,
        )

    def _build_config(self) -> dict[str, Any]:
        return {
            "min_df": self.min_df,
            "max_df": self.max_df,
            "sublinear_tf": self.sublinear_tf,
            "top_keywords": self.top_keywords,
        }

    # ---- query-time re-binding ---------------------------------------

    def rebuild_vectorizer(
        self,
        analyzer: AnalyzerCallable,
        vocabulary: dict[str, int],
        idf: np.ndarray,
        config: dict[str, Any],
    ) -> TfidfVectorizer:
        """保存された語彙+IDF+config から fresh TfidfVectorizer を再構成."""
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            min_df=config.get("min_df", 1),
            max_df=config.get("max_df", 0.95),
            sublinear_tf=config.get("sublinear_tf", True),
            vocabulary=vocabulary,
        )
        # vocabulary を注入した後で idf を注入するため、空文書に対して一度 fit
        # (TfidfVectorizer は vocabulary 指定で fit せず呼べないため dummy corpus で済ませる)
        vectorizer.fit([" ".join(vocabulary.keys())])
        # scikit-learn の内部で _tfidf.idf_ を使うので上書き
        vectorizer.idf_ = idf
        return vectorizer

    def transform_query(
        self,
        vectorizer: TfidfVectorizer,
        query_text: str,
    ) -> csr_matrix:
        vec = vectorizer.transform([query_text])
        if not isinstance(vec, csr_matrix):
            vec = csr_matrix(vec)
        return vec
