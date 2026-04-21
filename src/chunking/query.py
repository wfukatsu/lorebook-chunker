"""`query` CLI: load artifacts + cosine top-K ベースライン検索."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

from chunking.normalize import normalize_text
from chunking.tfidf import TfidfBuilder


class QueryAnalyzer(Protocol):
    def tokenize_for_tfidf(self, text: str) -> list[str]: ...


@dataclass
class QueryHit:
    chunk_id: str
    score: float
    source: str
    text: str
    top_keywords: list[dict[str, Any]]


@dataclass
class QueryResult:
    exit_code: int
    hits: list[QueryHit]
    error_message: str = ""


def run_query_impl(
    output_dir: Path,
    query_text: str,
    *,
    top_k: int,
    analyzer_factory=None,
) -> QueryResult:
    """テスト可能な純粋関数. CLI 依存なし."""
    if analyzer_factory is None:
        from chunking.ingest import default_analyzer_factory
        analyzer_factory = default_analyzer_factory

    vocab_path = output_dir / "vocab.npz"
    chunks_path = output_dir / "chunks.jsonl"
    analyzer_path = output_dir / "analyzer.json"

    missing = [p for p in (vocab_path, chunks_path, analyzer_path) if not p.exists()]
    if missing:
        return QueryResult(
            exit_code=2,
            hits=[],
            error_message=(
                f"必要な成果物が見つかりません: {', '.join(str(p) for p in missing)}。"
                "先に `ingest` を実行してください。"
            ),
        )

    # Load
    try:
        analyzer = analyzer_factory(analyzer_path)
    except Exception as e:
        return QueryResult(
            exit_code=3,
            hits=[],
            error_message=f"analyzer 復元に失敗しました ({e})。ingest 時と異なる環境で実行していませんか？",
        )

    try:
        arts = TfidfBuilder.load(vocab_path)
    except Exception as e:
        return QueryResult(exit_code=3, hits=[], error_message=f"vocab.npz 読み込み失敗: {e}")

    chunks = list(_load_chunks(chunks_path))
    if not chunks:
        return QueryResult(exit_code=3, hits=[], error_message="chunks.jsonl が空です")
    if len(chunks) != arts.matrix.shape[0]:
        return QueryResult(
            exit_code=3,
            hits=[],
            error_message=(
                f"chunks.jsonl ({len(chunks)}) と vocab.npz matrix ({arts.matrix.shape[0]} rows) が不整合"
            ),
        )

    # tokenize + transform
    builder = TfidfBuilder(
        analyzer=analyzer.tokenize_for_tfidf,
        min_df=arts.config.get("min_df", 1),
        max_df=arts.config.get("max_df", 1.0),
        sublinear_tf=arts.config.get("sublinear_tf", True),
    )
    vectorizer = builder.rebuild_vectorizer(
        analyzer.tokenize_for_tfidf, arts.vocabulary, arts.idf, arts.config
    )
    q_norm = normalize_text(query_text)
    q_vec = builder.transform_query(vectorizer, q_norm)
    if q_vec.nnz == 0:
        # 全語彙にヒットしない — 低スコア返却 (lint 相当の警告は stderr で別途)
        return QueryResult(
            exit_code=0,
            hits=[],
            error_message="クエリが既存語彙にヒットしませんでした (top_keywords 候補外)",
        )

    # Cosine similarity (matrix はすでに L2-normalized by TfidfVectorizer.norm='l2' default)
    # matrix @ q.T の値はコサイン類似度に一致する (TfidfVectorizer デフォルト)
    scores = (arts.matrix @ q_vec.T).toarray().flatten()
    k = min(top_k, len(scores))
    top_indices = np.argpartition(-scores, k - 1)[:k]
    top_sorted = top_indices[np.argsort(-scores[top_indices])]

    hits = [
        QueryHit(
            chunk_id=chunks[idx]["chunk_id"],
            score=float(scores[idx]),
            source=chunks[idx]["source"],
            text=chunks[idx]["text"],
            top_keywords=chunks[idx].get("top_keywords", []),
        )
        for idx in top_sorted
        if scores[idx] > 0
    ]
    return QueryResult(exit_code=0, hits=hits)


def _load_chunks(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_query(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    result = run_query_impl(out, args.query_text, top_k=int(args.top_k))
    if result.exit_code != 0:
        print(f"[error] {result.error_message}", file=sys.stderr)
        return result.exit_code
    if result.error_message:
        print(f"[warn] {result.error_message}", file=sys.stderr)
    if not result.hits:
        print("(no matching chunks)", file=sys.stderr)
        return 0
    for i, hit in enumerate(result.hits, start=1):
        snippet = hit.text[:200].replace("\n", " ")
        keywords = ", ".join(kw["term"] for kw in hit.top_keywords[:5])
        print(f"\n[{i}] chunk_id={hit.chunk_id} score={hit.score:.4f} source={hit.source}")
        print(f"    keywords: {keywords}")
        print(f"    text: {snippet}")
    return 0
