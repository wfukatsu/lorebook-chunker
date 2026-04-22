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

from lorebook_chunker.schema import KeywordEntry

# analyzer callable 契約: str -> list[str]
AnalyzerCallable = Callable[[str], list[str]]


def _identity_analyzer(tokens: list[str]) -> list[str]:
    """pre-tokenized 入力用の恒等関数. モジュールトップレベルで定義し pickle 可能にしておく."""
    return list(tokens)


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
        max_df: int | float = 1.0,
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

    def fit_transform_pretokenized(
        self, tokens_per_chunk: Sequence[Sequence[str]]
    ) -> tuple[csr_matrix, dict[str, int], np.ndarray]:
        """事前トークナイズ済み入力の高速パス.

        ingest 側で chunk ごとの Doc を 1 度だけ計算した後、そのトークン列を直接
        投入するために使う. sklearn に identity analyzer を注入することで
        TfidfVectorizer の analyzer(text) 経由の GiNZA 再呼び出しを完全に回避する.

        返り値とスキーマは fit_transform と同一 (matrix, vocabulary, idf).
        """
        if not tokens_per_chunk:
            raise ValueError("fit_transform_pretokenized: tokens_per_chunk must be non-empty")
        # sklearn の analyzer 契約は callable(input) -> list[str]. 事前トークナイズ済みなら
        # input はすでに list[str] なので恒等関数で流す. lowercase/stop_words の前処理も
        # skip したいので analyzer ルートを使う (tokenizer=... は lowercase が噛む).
        self._vectorizer = TfidfVectorizer(
            analyzer=_identity_analyzer,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
        )
        # TfidfVectorizer は入力 iterable を 1 度だけ走査するので、
        # 事前に全体を list に展開するコピーは冗長 (数 MB〜数十 MB の無駄).
        # list[str] は identity_analyzer が pass-through するので Sequence のまま渡せる.
        matrix = self._vectorizer.fit_transform(tokens_per_chunk)
        if not isinstance(matrix, csr_matrix):
            matrix = csr_matrix(matrix)
        vocabulary: dict[str, int] = dict(self._vectorizer.vocabulary_)
        idf = np.asarray(self._vectorizer.idf_, dtype=np.float64)
        return matrix, vocabulary, idf

    def top_keywords_per_chunk(
        self, matrix: csr_matrix, vocabulary: dict[str, int]
    ) -> list[list[KeywordEntry]]:
        """各行の TF-IDF 上位 N 語を返す.

        F-039: CSR の data/indices/indptr を直接走査し、`argpartition` で上位 K を抽出.
        """
        # idx -> term の配列を 1 度だけ構築 (O(V)).
        inv_vocab_arr: list[str] = [""] * len(vocabulary)
        for term, idx in vocabulary.items():
            inv_vocab_arr[idx] = term

        data = matrix.data
        indices = matrix.indices
        indptr = matrix.indptr
        rows: list[list[KeywordEntry]] = []
        top_k = self.top_keywords
        for row_idx in range(matrix.shape[0]):
            start = indptr[row_idx]
            end = indptr[row_idx + 1]
            row_data = data[start:end]
            row_indices = indices[start:end]
            if row_data.size == 0:
                rows.append([])
                continue
            k = min(top_k, row_data.size)
            # argpartition + argsort で top-k を降順に取り出す.
            partitioned = np.argpartition(-row_data, k - 1)[:k]
            ordered = partitioned[np.argsort(-row_data[partitioned])]
            rows.append(
                [
                    {
                        "term": inv_vocab_arr[int(row_indices[i])],
                        "tfidf": float(row_data[i]),
                    }
                    for i in ordered
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
        """vocab.npz に matrix + vocabulary + idf + config を統合保存.

        F-002 + F-026: `object` dtype は np.load(allow_pickle=True) を要求し、
        悪意ある .npz が任意コード実行を許す経路になる. vocabulary_terms は
        `np.str_` (=unicode) dtype、config は utf-8 エンコード済みバイト列で保存し、
        load 側は `allow_pickle=False` で読めるようにする.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = self._build_config()
        # vocabulary は idx 順の terms 配列で保存 (dict は np.save できないため)
        terms_in_order = [""] * len(vocabulary)
        for term, idx in vocabulary.items():
            terms_in_order[idx] = term
        vocab_array = np.array(terms_in_order, dtype=np.str_)
        config_bytes = json.dumps(config, ensure_ascii=False).encode("utf-8")
        np.savez_compressed(
            path,
            matrix_data=matrix.data,
            matrix_indices=matrix.indices,
            matrix_indptr=matrix.indptr,
            matrix_shape=np.asarray(matrix.shape, dtype=np.int64),
            vocabulary_terms=vocab_array,
            idf=idf,
            config_json=np.frombuffer(config_bytes, dtype=np.uint8),
        )

    @staticmethod
    def load(path: str | Path) -> TfidfArtifacts:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            matrix = csr_matrix(
                (
                    data["matrix_data"],
                    data["matrix_indices"],
                    data["matrix_indptr"],
                ),
                shape=tuple(data["matrix_shape"]),
            )
            terms_raw = data["vocabulary_terms"]
            vocabulary = {str(term): idx for idx, term in enumerate(terms_raw)}
            idf = np.asarray(data["idf"], dtype=np.float64)
            # config_json は uint8 配列で保存されているので bytes に戻して JSON decode.
            config_bytes = bytes(np.asarray(data["config_json"], dtype=np.uint8).tobytes())
            config = json.loads(config_bytes.decode("utf-8"))
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
        # (TfidfVectorizer は vocabulary 指定で fit せず呼べないため dummy corpus で済ませる).
        # 以前は " ".join(vocabulary.keys()) を渡していたが、大規模コーパスで語彙数が
        # 数千を超えると analyzer (=GiNZA pipeline) が巨大文字列を parse して
        # bunsetu_recognizer で RecursionError を起こした. idf_ は直後に上書きするため
        # ダミーは空文書で良い.
        vectorizer.fit([""])
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
