"""`query` CLI: load artifacts + cosine top-K ベースライン検索."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lorebook_chunker._io import load_chunks
from lorebook_chunker.normalize import normalize_text
from lorebook_chunker.schema import ChunkFileCorruptError
from lorebook_chunker.tfidf import TfidfBuilder


def positive_int(value: str) -> int:
    """argparse 用 custom type: 正の整数のみ受け付ける.

    F-062: ``--top-k 0`` や負の値は ``--top-k must be a positive integer`` で reject.
    """
    try:
        parsed = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--top-k must be a positive integer (got {value!r})"
        ) from e
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--top-k must be a positive integer")
    return parsed


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
        from lorebook_chunker.ingest import default_analyzer_factory
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

    try:
        chunks = list(load_chunks(chunks_path))
    except ChunkFileCorruptError as e:
        return QueryResult(exit_code=3, hits=[], error_message=str(e))
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
    # F-038: クエリトークン 0 or 全 OOV は exit 4 (no-match).
    q_tokens = analyzer.tokenize_for_tfidf(q_norm)
    if not q_tokens:
        return QueryResult(
            exit_code=4,
            hits=[],
            error_message="クエリをトークン化した結果が空でした (全てストップワード?)",
        )
    q_vec = builder.transform_query(vectorizer, q_norm)
    if q_vec.nnz == 0:
        return QueryResult(
            exit_code=4,
            hits=[],
            error_message="クエリが既存語彙にヒットしませんでした (top_keywords 候補外)",
        )

    # Cosine similarity (matrix はすでに L2-normalized by TfidfVectorizer.norm='l2' default)
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
    if not hits:
        # F-038: スコア > 0 のヒットなし = 実質 OOV.
        return QueryResult(
            exit_code=4,
            hits=[],
            error_message="no hits above zero score",
        )
    return QueryResult(exit_code=0, hits=hits)


def run_query(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    result = run_query_impl(out, args.query_text, top_k=int(args.top_k))
    fmt = getattr(args, "format", None) or ("json" if not sys.stdout.isatty() else "human")
    if result.exit_code != 0:
        print(f"[error] {result.error_message}", file=sys.stderr)
        if fmt == "json":
            print(json.dumps({"error": result.error_message, "hits": []}, ensure_ascii=False))
        return result.exit_code
    if result.error_message:
        print(f"[warn] {result.error_message}", file=sys.stderr)
    if fmt == "json":
        # F-017: 機械可読出力. stdout に JSON、ヒューマンログは stderr.
        payload = [
            {
                "rank": i + 1,
                "chunk_id": h.chunk_id,
                "score": h.score,
                "source": h.source,
                "text": h.text,
                "top_keywords": h.top_keywords,
            }
            for i, h in enumerate(result.hits)
        ]
        print(json.dumps(payload, ensure_ascii=False))
        return 0
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
